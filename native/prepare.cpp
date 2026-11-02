// Offline preparation only. LibTorch is never linked into the inference plugin.
#include <torch/torch.h>
#include <torch/serialize.h>
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <regex>
#include <map>
#include <filesystem>
#include <numeric>
#include <cmath>
#ifdef DESKPET_PREP_ORT
#include "onnxruntime_cxx_api.h"
#endif
using json=nlohmann::json;
static void check(bool ok,const std::string &e){if(!ok)throw std::runtime_error(e);}
static json readjson(const std::string &p){std::ifstream f(p);check(bool(f),"Missing file: "+p);return json::parse(f);}
static std::map<std::string,torch::Tensor> safetensors(const std::string &path) {
    std::ifstream f(path,std::ios::binary);uint64_t size=0;f.read(reinterpret_cast<char *>(&size),8);check(size<100*1024*1024 && bool(f),"Invalid safetensors header");
    std::string header(size,'\0');f.read(header.data(),size);auto j=json::parse(header);std::map<std::string,torch::Tensor> tensors;
    auto fileSize=std::filesystem::file_size(path);
    for(auto it=j.begin();it!=j.end();++it){if(it.key()=="__metadata__")continue;auto &v=it.value();auto offsets=v.at("data_offsets").get<std::vector<uint64_t>>();
        check(offsets.size()==2 && offsets[1]>=offsets[0] && offsets[1]<=fileSize-8-size,"Invalid tensor offsets");
        auto shape=v.at("shape").get<std::vector<int64_t>>();std::string dtype=v.at("dtype");auto type=dtype=="F32"?torch::kFloat32:dtype=="F16"?torch::kFloat16:dtype=="I64"?torch::kInt64:torch::kBFloat16;
        check(dtype=="F32"||dtype=="F16"||dtype=="BF16"||dtype=="I64","Unsupported tensor dtype: "+dtype);
        auto tensor=torch::empty(shape,torch::TensorOptions().dtype(type));check(tensor.nbytes()==offsets[1]-offsets[0],"Tensor size mismatch");
        f.seekg(8+size+offsets[0]);f.read(static_cast<char *>(tensor.data_ptr()),tensor.nbytes());check(bool(f),"Truncated tensor");tensors[it.key()]=std::move(tensor);
    }return tensors;
}
static void writereference(const std::string &base,const std::string &adapter,const std::string &adapterConfig,const std::string &output) {
    auto tensors=safetensors(base),lora=safetensors(adapter);auto cfg=readjson(adapterConfig);
    check(!cfg.value("use_dora",false)&&!cfg.value("use_rslora",false)&&cfg.value("bias",std::string("none"))=="none","Unsupported LoRA variant");
    size_t applied=0;
    for(const auto &entry:lora) {
        auto pos=entry.first.find(".lora_A");if(pos==std::string::npos)continue;
        auto key=entry.first.substr(0,pos)+".weight";
        const std::string prefix="base_model.model.";if(key.rfind(prefix,0)==0)key=key.substr(prefix.size());
        if(!tensors.count(key)) {
            if(tensors.count("model.language_model."+key))key="model.language_model."+key;
            else if(tensors.count("model."+key))key="model."+key;
        }
        auto found=tensors.find(key);check(found!=tensors.end(),"Missing reference weight: "+key);
        auto bname=std::regex_replace(entry.first,std::regex("lora_A"),"lora_B");check(lora.count(bname),"Missing LoRA B");
        auto delta=torch::matmul(lora.at(bname).to(torch::kFloat32),entry.second.to(torch::kFloat32))*(cfg.at("lora_alpha").get<double>()/cfg.at("r").get<double>());
        check(delta.sizes()==found->second.sizes(),"Reference LoRA shape mismatch");
        found->second=(found->second.to(torch::kFloat32)+delta).to(found->second.scalar_type()).contiguous();++applied;
    }
    check(applied>0&&applied*2==lora.size(),"Not every LoRA module was merged into reference");
    json header=json::object();uint64_t offset=0;
    for(auto &entry:tensors) {
        auto t=entry.second.contiguous();entry.second=t;std::string dtype=t.scalar_type()==torch::kFloat32?"F32":t.scalar_type()==torch::kFloat16?"F16":t.scalar_type()==torch::kInt64?"I64":"BF16";
        header[entry.first]={{"dtype",dtype},{"shape",t.sizes().vec()},{"data_offsets",{offset,offset+t.nbytes()}}};offset+=t.nbytes();
    }
    header["__metadata__"]={{"format","pt"},{"deskpet_lora_modules",std::to_string(applied)}};
    auto text=header.dump();while(text.size()%8)text+=' ';uint64_t length=text.size();
    std::ofstream out(output,std::ios::binary);out.write(reinterpret_cast<char *>(&length),8);out.write(text.data(),text.size());
    for(const auto &entry:tensors)out.write(static_cast<const char *>(entry.second.data_ptr()),entry.second.nbytes());
    check(bool(out),"Failed writing native reference safetensors");std::cout<<"Reference safetensors LoRA modules: "<<applied<<"\n";
}
static void writeheads(const std::string &input,const std::string &output) {
    std::ifstream f(input,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(f)),{});check(!bytes.empty(),"Empty heads checkpoint");
    auto state=torch::pickle_load(bytes).toGenericDict();auto ctx=ggml_init({4*1024*1024,nullptr,true});auto meta=gguf_init_empty();
    std::vector<torch::Tensor> storage;gguf_set_val_str(meta,"general.architecture","deskpet-router-heads");
    for(auto &entry:state){auto name=entry.key().toStringRef();auto tensor=entry.value().toTensor();
        if(name=="confidence_calibrated")continue;
        check(name.rfind("backbone.",0)!=0,"Backbone tensor found in heads checkpoint");
        auto half=tensor.to(torch::kFloat16).contiguous();std::vector<int64_t> dims(half.sizes().begin(),half.sizes().end());
        check(dims.size()>=1&&dims.size()<=4,"Unsupported head rank");std::reverse(dims.begin(),dims.end());
        auto t=ggml_new_tensor(ctx,GGML_TYPE_F16,int(dims.size()),dims.data());ggml_set_name(t,name.c_str());t->data=half.data_ptr();storage.push_back(half);gguf_add_tensor(meta,t);
    }
    check(gguf_write_to_file(meta,output.c_str(),false),"Failed writing heads");gguf_free(meta);ggml_free(ctx);
}
static void inspect(const std::string &file) {
    ggml_context *ctx=nullptr;auto meta=gguf_init_from_file(file.c_str(),{true,&ctx});check(meta&&ctx,"Invalid GGUF");
    uint64_t bytes=0;json types=json::object();
    for(int64_t i=0;i<gguf_get_n_tensors(meta);++i) {
        auto tensor=ggml_get_tensor(ctx,gguf_get_tensor_name(meta,i));auto size=ggml_nbytes(tensor);bytes+=size;
        check(gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,i)+size<=std::filesystem::file_size(file),"Truncated GGUF tensor");
        auto name=std::string(ggml_type_name(tensor->type));types[name]=types.value(name,0)+1;
    }
    std::cout<<json{{"tensors",gguf_get_n_tensors(meta)},{"tensorBytes",bytes},{"types",types}}.dump(2)<<"\n";gguf_free(meta);ggml_free(ctx);
}
static void audittokenizer(const std::string &input,const std::string &file) {
    auto tokenizer=readjson(input);auto meta=gguf_init_from_file(file.c_str(),{true,nullptr});check(meta,"Invalid GGUF");
    auto key=gguf_find_key(meta,"tokenizer.ggml.tokens");check(key>=0,"Missing vocabulary");auto count=gguf_get_arr_n(meta,key);size_t verified=0;
    auto token=[&](const std::string &text,int64_t id){check(id>=0&&uint64_t(id)<count,"Token ID outside GGUF vocabulary");check(text==gguf_get_arr_str(meta,key,id),"Tokenizer ID mismatch: "+std::to_string(id));++verified;};
    auto &vocab=tokenizer.at("model").at("vocab");for(auto it=vocab.begin();it!=vocab.end();++it)token(it.key(),it.value().get<int64_t>());
    for(const auto &added:tokenizer.at("added_tokens"))token(added.at("content"),added.at("id"));
    std::cout<<"Tokenizer entries verified: "<<verified<<"; GGUF vocabulary: "<<count<<"\n";gguf_free(meta);
}
static void auditheads(const std::string &input,const std::string &file) {
    std::ifstream source(input,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(source)),{});
    auto state=torch::pickle_load(bytes).toGenericDict();ggml_context *ctx=nullptr;auto meta=gguf_init_from_file(file.c_str(),{true,&ctx});check(meta&&ctx,"Invalid heads GGUF");
    std::ifstream f(file,std::ios::binary);int64_t checked=0;
    for(const auto &entry:state) {
        auto name=entry.key().toStringRef();if(name=="confidence_calibrated")continue;
        auto value=entry.value().toTensor().to(torch::kFloat16).contiguous();auto i=gguf_find_tensor(meta,name.c_str());check(i>=0,"Missing head: "+name);
        auto tensor=ggml_get_tensor(ctx,name.c_str());check(tensor->type==GGML_TYPE_F16&&size_t(value.nbytes())==ggml_nbytes(tensor),"Head dtype/size mismatch");
        for(size_t d=0;d<value.dim();++d)check(value.size(d)==tensor->ne[value.dim()-1-d],"Head shape mismatch");
        std::vector<char> actual(value.nbytes());f.seekg(gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,i));f.read(actual.data(),actual.size());
        check(size_t(f.gcount())==actual.size()&&std::memcmp(actual.data(),value.data_ptr(),actual.size())==0,"Head conversion mismatch: "+name);++checked;
    }
    check(checked==gguf_get_n_tensors(meta),"Unexpected exported head");std::cout<<"FP16 head tensors verified: "<<checked<<"; exact cast-byte match\n";gguf_free(meta);ggml_free(ctx);
}
#ifdef DESKPET_PREP_ORT
static void auditemotion(const std::string &weights,const std::string &config,const std::string &onnx) {
    auto w=safetensors(weights);auto cfg=readjson(config);int hidden=cfg.at("hidden_size"),heads=cfg.at("num_attention_heads"),layers=cfg.at("num_hidden_layers");
    double epsilon=cfg.at("layer_norm_eps");auto ids=torch::zeros({1,128},torch::kInt64);auto mask=torch::zeros({1,128},torch::kInt64),types=torch::zeros_like(ids);
    for(int i=0;i<19;++i){ids[0][i]=i==0?2:i==18?3:100+i;mask[0][i]=1;}
    auto linear=[&](torch::Tensor x,const std::string &p){return torch::matmul(x,w.at(p+".weight").t())+w.at(p+".bias");};
    auto norm=[&](torch::Tensor x,const std::string &p){return torch::layer_norm(x,{x.size(-1)},w.at(p+".weight"),w.at(p+".bias"),epsilon);};
    auto gather=[&](const std::string &p,torch::Tensor index){return w.at(p).index_select(0,index.flatten()).reshape({1,128,-1});};
    auto x=norm(gather("electra.embeddings.word_embeddings.weight",ids)+gather("electra.embeddings.position_embeddings.weight",torch::arange(128,torch::kInt64))+gather("electra.embeddings.token_type_embeddings.weight",types),"electra.embeddings.LayerNorm");
    if(w.count("electra.embeddings_project.weight"))x=linear(x,"electra.embeddings_project");
    auto bias=(1-mask.to(torch::kFloat32)).reshape({1,1,1,128})*-10000;
    for(int i=0;i<layers;++i){auto p="electra.encoder.layer."+std::to_string(i);auto project=[&](const std::string &key){return linear(x,p+".attention.self."+key).reshape({1,128,heads,hidden/heads}).permute({0,2,1,3});};
        auto q=project("query"),k=project("key"),v=project("value");auto scores=torch::matmul(q,k.transpose(-1,-2))/std::sqrt(double(hidden/heads))+bias;
        auto attended=torch::matmul(torch::softmax(scores,-1),v).permute({0,2,1,3}).contiguous().reshape({1,128,hidden});
        x=norm(linear(attended,p+".attention.output.dense")+x,p+".attention.output.LayerNorm");
        x=norm(linear(torch::gelu(linear(x,p+".intermediate.dense")),p+".output.dense")+x,p+".output.LayerNorm");
    }
    auto reference=linear(torch::gelu(linear(x.select(1,0),"classifier.dense")),"classifier.out_proj");
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"conversion-audit"};Ort::SessionOptions options;options.SetIntraOpNumThreads(2);Ort::Session session(env,onnx.c_str(),options);
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);int64_t shape[]={1,128};std::vector<Ort::Value> inputs;
    for(auto &tensor:std::vector<torch::Tensor>{ids,mask,types})inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory,tensor.data_ptr<int64_t>(),128,shape,2));
    const char *names[]={"input_ids","attention_mask","token_type_ids"};Ort::AllocatorWithDefaultOptions allocator;auto name=session.GetOutputNameAllocated(0,allocator);const char *outputs[]={name.get()};
    auto result=session.Run(Ort::RunOptions{nullptr},names,inputs.data(),3,outputs,1);auto values=result[0].GetTensorData<float>();
    double error=0;for(int i=0;i<2;++i)error=std::max(error,std::abs(double(values[i]-reference[0][i].item<float>())));
    check(error<1e-4,"ONNX conversion logits mismatch");std::cout<<"ONNX / LibTorch reference maximum logit error: "<<error<<"\n";
}
#endif
static torch::Tensor reorder(torch::Tensor t,int axis,int nk,int nv,int dim) {
    auto shape=t.sizes().vec();auto expanded=shape;expanded.erase(expanded.begin()+axis);expanded.insert(expanded.begin()+axis,{nk,nv/nk,dim});
    auto perm=std::vector<int64_t>(expanded.size());std::iota(perm.begin(),perm.end(),0);std::swap(perm[axis],perm[axis+1]);
    return t.reshape(expanded).permute(perm).contiguous().reshape(shape);
}
static void merge(const std::string &base,const std::string &adapter,const std::string &adapterConfig,const std::string &baseConfig,const std::string &output) {
    auto lora=safetensors(adapter);auto cfg=readjson(adapterConfig),h=readjson(baseConfig);if(h.contains("text_config"))h=h["text_config"];
    check(!cfg.value("use_dora",false)&&!cfg.value("use_rslora",false)&&cfg.value("bias",std::string("none"))=="none","Unsupported LoRA variant");
    auto ctx=static_cast<ggml_context *>(nullptr);auto meta=gguf_init_from_file(base.c_str(),{false,&ctx});check(meta&&ctx,"Invalid GGUF base");
    check(gguf_get_val_u32(meta,gguf_find_key(meta,"general.file_type"))==uint32_t(LLAMA_FTYPE_MOSTLY_BF16),"Base must be the unquantized BF16 reference");
    const std::map<std::string,std::string> names={{"self_attn.q_proj","attn_q"},{"self_attn.k_proj","attn_k"},{"self_attn.v_proj","attn_v"},{"self_attn.o_proj","attn_output"},
        {"mlp.up_proj","ffn_up"},{"mlp.gate_proj","ffn_gate"},{"mlp.down_proj","ffn_down"},{"linear_attn.in_proj_qkv","attn_qkv"},
        {"linear_attn.in_proj_z","attn_gate"},{"linear_attn.in_proj_a","ssm_alpha"},{"linear_attn.in_proj_b","ssm_beta"},{"linear_attn.out_proj","ssm_out"}};
    std::regex pattern(".*layers\\.([0-9]+)\\.(.*)\\.lora_A(?:\\.default)?\\.weight");size_t applied=0;
    for(auto &entry:lora){std::smatch match;if(!std::regex_match(entry.first,match,pattern))continue;
        auto name=match[2].str();check(names.count(name),"Unmapped LoRA module: "+name);
        auto bname=std::regex_replace(entry.first,std::regex("lora_A"),"lora_B");check(lora.count(bname),"Missing LoRA B");
        auto delta=torch::matmul(lora[bname].to(torch::kFloat32),entry.second.to(torch::kFloat32))*(cfg.at("lora_alpha").get<double>()/cfg.at("r").get<double>());
        int nk=h.value("linear_num_key_heads",0),nv=h.value("linear_num_value_heads",0),dim=h.value("linear_value_head_dim",128);
        if(nk&&nv!=nk){
            if(name=="linear_attn.in_proj_qkv"){int qk=2*nk*h.value("linear_key_head_dim",128);delta=torch::cat({delta.slice(0,0,qk),reorder(delta.slice(0,qk),0,nk,nv,dim)},0);}
            else if(name=="linear_attn.in_proj_z")delta=reorder(delta,0,nk,nv,dim);
            else if(name=="linear_attn.in_proj_a"||name=="linear_attn.in_proj_b")delta=reorder(delta,0,nk,nv,1);
            else if(name=="linear_attn.out_proj")delta=reorder(delta,1,nk,nv,dim);
        }
        auto ggname="blk."+match[1].str()+"."+names.at(name)+".weight";auto tensor=ggml_get_tensor(ctx,ggname.c_str());check(tensor,"Missing GGUF tensor: "+ggname);
        check(tensor->type==GGML_TYPE_BF16,"Merged tensor must be BF16");
        auto original=torch::from_blob(tensor->data,{tensor->ne[1],tensor->ne[0]},torch::kBFloat16);
        check(original.sizes()==delta.sizes(),"LoRA shape mismatch");original.copy_((original.to(torch::kFloat32)+delta).to(torch::kBFloat16));++applied;
    }
    check(applied*2==lora.size(),"Not every adapter tensor was applied");check(applied>0,"No LoRA tensors applied");
    // Build writer tensor descriptors from the allocated context. Reader
    // metadata descriptors are only shapes/offsets, not allocated tensors.
    auto outputMeta=gguf_init_empty();gguf_set_kv(outputMeta,meta);
    for(int64_t i=0;i<gguf_get_n_tensors(meta);++i) {
        auto name=gguf_get_tensor_name(meta,i);auto tensor=ggml_get_tensor(ctx,name);
        check(tensor&&tensor->data,"Missing base tensor data");gguf_add_tensor(outputMeta,tensor);
    }
    gguf_set_val_u32(outputMeta,"deskpet.lora_modules",uint32_t(applied));
    check(gguf_write_to_file(outputMeta,output.c_str(),false),"Failed writing merged backbone");gguf_free(outputMeta);gguf_free(meta);ggml_free(ctx);
    std::cout<<"Merged LoRA modules: "<<applied<<"\n";
}
// Minimal protobuf writer for a standard ONNX opset-17 Electra graph. Only model
// conversion lives here: execution is delegated to the unmodified ORT C++ API.
namespace pb {
using S=std::string;
static S var(uint64_t n){S s;do{auto b=uint8_t(n&127);n>>=7;s+=char(b|(n?128:0));}while(n);return s;}
static S i(int key,int64_t n){return var(uint64_t(key)<<3)+var(uint64_t(n));}
static S s(int key,const S &v){return var((uint64_t(key)<<3)|2)+var(v.size())+v;}
static S fl(int key,float f){S x(reinterpret_cast<char *>(&f),4);return var((uint64_t(key)<<3)|5)+x;}
static S info(const S &name,int type,const std::vector<int64_t> &shape){S dims;for(auto d:shape)dims+=s(1,i(1,d));return s(1,name)+s(2,s(1,i(1,type)+s(2,dims)));}
static S tensor(const S &name,torch::Tensor t){t=t.contiguous();S dims;for(auto d:t.sizes())dims+=i(1,d);return dims+i(2,t.scalar_type()==torch::kInt64?7:1)+s(8,name)+s(9,S(static_cast<const char *>(t.data_ptr()),t.nbytes()));}
static S attr(const S &name,int64_t value){return s(1,name)+i(3,value)+i(20,2);}
static S attrs(const S &name,std::vector<int64_t> values){S x=s(1,name);for(auto v:values)x+=i(8,v);return x+i(20,7);}
struct Graph {
 S nodes,initializers;int counter=0;std::map<S,torch::Tensor> weights;
 explicit Graph(std::map<S,torch::Tensor> w):weights(std::move(w)){for(auto &e:weights)initializers+=s(5,tensor(e.first,e.second.to(torch::kFloat32)));}
 S constant(torch::Tensor t){auto n="constant_"+std::to_string(++counter);initializers+=s(5,tensor(n,t));return n;}
 S op(const S &kind,std::vector<S> inputs,const S &attributes=""){auto name="node_"+std::to_string(++counter);S n;for(auto &in:inputs)n+=s(1,in);n+=s(2,name)+s(4,kind)+attributes;nodes+=s(1,n);return name;}
 S integer(const S &kind,const S &x,const S &key,int64_t v){return op(kind,{x},s(5,attr(key,v)));}
 S reshape(const S &x,std::vector<int64_t> dims){return op("Reshape",{x,constant(torch::tensor(dims,torch::kInt64))});}
 S transpose(const S &x,std::vector<int64_t> dims){return op("Transpose",{x},s(5,attrs("perm",dims)));}
 S linear(const S &x,const S &prefix){return op("Add",{op("MatMul",{x,transpose(prefix+".weight",{1,0})}),prefix+".bias"});}
 S norm(const S &x,const S &prefix,float epsilon){return op("LayerNormalization",{x,prefix+".weight",prefix+".bias"},s(5,attr("axis",-1))+s(5,s(1,"epsilon")+fl(2,epsilon)+i(20,1)));}
 S scalar(float f){return constant(torch::tensor(f,torch::kFloat32));}
 S gelu(const S &x){return op("Mul",{op("Mul",{x,scalar(.5f)}),op("Add",{scalar(1),op("Erf",{op("Div",{x,scalar(std::sqrt(2.0f))})})})});}
};
}
static void emotion(const std::string &weights,const std::string &config,const std::string &output) {
    auto cfg=readjson(config);check(cfg.at("model_type")=="electra","Expected Electra configuration");
    auto tensors=safetensors(weights);pb::Graph g(std::move(tensors));int hidden=cfg.at("hidden_size"),heads=cfg.at("num_attention_heads"),layers=cfg.at("num_hidden_layers");float epsilon=cfg.at("layer_norm_eps");
    auto gather=[&](std::string table,std::string indices){return g.op("Gather",{table,indices},pb::s(5,pb::attr("axis",0)));};
    auto positions=g.constant(torch::arange(128,torch::kInt64).reshape({1,128}));
    auto x=g.op("Add",{g.op("Add",{gather("electra.embeddings.word_embeddings.weight","input_ids"),gather("electra.embeddings.position_embeddings.weight",positions)}),gather("electra.embeddings.token_type_embeddings.weight","token_type_ids")});
    x=g.norm(x,"electra.embeddings.LayerNorm",epsilon);
    if(g.weights.count("electra.embeddings_project.weight"))x=g.linear(x,"electra.embeddings_project");
    auto mask=g.op("Mul",{g.op("Sub",{g.scalar(1),g.integer("Cast","attention_mask","to",1)}),g.scalar(-10000)});mask=g.reshape(mask,{1,1,1,128});
    for(int i=0;i<layers;++i){auto prefix="electra.encoder.layer."+std::to_string(i);auto projection=[&](const std::string &kind){return g.transpose(g.reshape(g.linear(x,prefix+".attention.self."+kind),{1,128,heads,hidden/heads}),{0,2,1,3});};
        auto q=projection("query"),k=projection("key"),v=projection("value");auto scores=g.op("Add",{g.op("Div",{g.op("MatMul",{q,g.transpose(k,{0,1,3,2})}),g.scalar(std::sqrt(float(hidden/heads)))}),mask});
        auto context=g.reshape(g.transpose(g.op("MatMul",{g.integer("Softmax",scores,"axis",-1),v}),{0,2,1,3}),{1,128,hidden});
        x=g.norm(g.op("Add",{g.linear(context,prefix+".attention.output.dense"),x}),prefix+".attention.output.LayerNorm",epsilon);
        x=g.norm(g.op("Add",{g.linear(g.gelu(g.linear(x,prefix+".intermediate.dense")),prefix+".output.dense"),x}),prefix+".output.LayerNorm",epsilon);
    }
    auto first=g.op("Gather",{x,g.constant(torch::tensor(std::vector<int64_t>{0},torch::kInt64))},pb::s(5,pb::attr("axis",1)));
    auto result=g.linear(g.gelu(g.linear(g.reshape(first,{1,hidden}),"classifier.dense")),"classifier.out_proj");
    auto graph=g.nodes+pb::s(2,"deskpet-koelectra-small")+g.initializers;
    for(auto name:{"input_ids","attention_mask","token_type_ids"})graph+=pb::s(11,pb::info(name,7,{1,128}));
    graph+=pb::s(12,pb::info(result,1,{1,2}));auto model=pb::i(1,8)+pb::s(2,"deskpet-cpp-preparation")+pb::s(7,graph)+pb::s(8,pb::i(2,17));
    std::ofstream out(output,std::ios::binary);out.write(model.data(),model.size());check(bool(out),"Failed writing emotion graph");
}
int main(int argc,char **argv){try{
    if(argc==3 && std::string(argv[1])=="inspect")inspect(argv[2]);
    else if(argc==4 && std::string(argv[1])=="audit-heads")auditheads(argv[2],argv[3]);
    else if(argc==4 && std::string(argv[1])=="audit-tokenizer")audittokenizer(argv[2],argv[3]);
#ifdef DESKPET_PREP_ORT
    else if(argc==5 && std::string(argv[1])=="audit-emotion")auditemotion(argv[2],argv[3],argv[4]);
#endif
    else if(argc==4 && std::string(argv[1])=="heads")writeheads(argv[2],argv[3]);
    else if(argc==4 && std::string(argv[1])=="quantize") {
        auto p=llama_model_quantize_default_params();p.ftype=LLAMA_FTYPE_MOSTLY_Q4_0;p.nthread=4;
        check(llama_model_quantize(argv[2],argv[3],&p)==0,"INT4 quantization failed");
    }
    else if(argc==7 && std::string(argv[1])=="merge")merge(argv[2],argv[3],argv[4],argv[5],argv[6]);
    else if(argc==6 && std::string(argv[1])=="merge-reference")writereference(argv[2],argv[3],argv[4],argv[5]);
    else if(argc==5 && std::string(argv[1])=="emotion")emotion(argv[2],argv[3],argv[4]);
    else throw std::runtime_error("Usage: heads input.pt output.gguf | merge base.gguf adapter.safetensors adapter_config.json base_config.json merged.gguf | emotion model.safetensors config.json model.onnx");
    return 0;
}catch(const std::exception &e){std::cerr<<"Preparation failed: "<<e.what()<<"\n";return 1;}}
