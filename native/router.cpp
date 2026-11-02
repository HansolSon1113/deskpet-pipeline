#include "runtime.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "chat.h"
#include "router-head-graph.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
using Tensor = ggml_tensor;
static std::pair<int,double> best(const std::vector<float> &v) {
    require(!v.empty(),"EMPTY_LOGITS"); int i=int(std::max_element(v.begin(),v.end())-v.begin());
    double sum=0; for(float f:v) sum+=std::exp(double(f-v[i])); return {i,1.0/sum};
}
static size_t characters(const std::string &s) { size_t n=0;for(unsigned char c:s) if((c&0xc0)!=0x80) ++n;return n; }
static size_t char_byte(const std::string &s,size_t c) {
    size_t n=0; for(size_t i=0;i<s.size();++i) if((uint8_t(s[i])&0xc0)!=0x80 && n++==c) return i;
    return s.size();
}
static std::pair<size_t,size_t> trimmed(const std::string &s) {
    auto white=[](unsigned char c){return c==' '||c=='\n'||c=='\r'||c=='\t'||c=='\f'||c=='\v';};
    size_t a=0,b=s.size();while(a<b&&white(s[a]))++a;while(b>a&&white(s[b-1]))--b;return {a,b};
}
struct HeadStore {
    ggml_context *weights=nullptr; ggml_backend_buffer_t buffer=nullptr; ggml_backend_t backend;
    explicit HeadStore(const std::string &file,ggml_backend_t b):backend(b) {
        gguf_init_params params{true,&weights}; auto meta=gguf_init_from_file(file.c_str(),params);
        require(meta && weights,"INVALID_HEADS_GGUF");
        try {
            for(int64_t i=0;i<gguf_get_n_tensors(meta);++i)
                require(gguf_get_tensor_type(meta,i)==GGML_TYPE_F16,"HEADS_MUST_BE_FP16");
            buffer=ggml_backend_alloc_ctx_tensors(weights,backend);require(buffer,"GPU_HEAD_ALLOCATION_FAILED");
            std::ifstream in(file,std::ios::binary);
            for(int64_t i=0;i<gguf_get_n_tensors(meta);++i) {
                auto t=ggml_get_tensor(weights,gguf_get_tensor_name(meta,i));
                size_t size=ggml_nbytes(t), offset=0; std::vector<char> bytes(std::min(size,size_t(8*1024*1024)));
                in.seekg(gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,i));
                while(offset<size) {auto count=std::min(bytes.size(),size-offset);in.read(bytes.data(),count);
                    require(size_t(in.gcount())==count,"TRUNCATED_HEADS");ggml_backend_tensor_set(t,bytes.data(),offset,count);offset+=count;}
            }
            gguf_free(meta);
        } catch(...) { gguf_free(meta);if(buffer)ggml_backend_buffer_free(buffer);ggml_free(weights);weights=nullptr;throw; }
    }
    ~HeadStore(){if(buffer)ggml_backend_buffer_free(buffer);if(weights)ggml_free(weights);}
    Tensor *get(const std::string &name) { auto t=ggml_get_tensor(weights,name.c_str()); require(t,"MISSING_HEAD: "+name);return t; }
    std::vector<float> calculate(const std::vector<float> &values,int hidden,int tokens,
        const std::function<Tensor *(ggml_context *,Tensor *)> &build) {
        auto ctx=ggml_init({16*1024*1024,nullptr,true});require(ctx,"GRAPH_ALLOCATION_FAILED");
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        try {
            auto input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,hidden,tokens);ggml_set_input(input);
            auto output=build(ctx,input);ggml_set_output(output);
            auto graph=ggml_new_graph_custom(ctx,4096,false);ggml_build_forward_expand(graph,output);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
                auto node=ggml_graph_node(graph,i);
                auto error="GPU_HEAD_OPERATION_UNSUPPORTED: "+std::string(ggml_op_name(node->op))+" "+ggml_get_name(node);
                if(node->src[0])error+=" src0="+std::string(ggml_type_name(node->src[0]->type));
                if(node->src[1])error+=" src1="+std::string(ggml_type_name(node->src[1]->type));
                require(ggml_backend_supports_op(backend,node),error);
            }
            require(ggml_gallocr_alloc_graph(alloc,graph),"GPU_HEAD_WORKSPACE_FAILED");
            require(values.size()==size_t(hidden*tokens),"INVALID_HIDDEN_STATE");
            ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"GPU_HEAD_COMPUTE_FAILED");
            std::vector<float> result(ggml_nelements(output));ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
            ggml_gallocr_free(alloc);ggml_free(ctx);return result;
        } catch(...) {ggml_gallocr_free(alloc);ggml_free(ctx);throw;}
    }
    Tensor *linear(ggml_context *ctx,Tensor *x,const std::string &prefix,bool bias=true) {
        auto y=deskpet_head_linear(ctx,x,get(prefix+".weight"),bias?get(prefix+".bias"):nullptr);
        ggml_set_name(y,prefix.c_str());return y;
    }
};

class LanguageModel final : public Model {
    std::unique_ptr<llama_model,decltype(&llama_model_free)> weights{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> context{nullptr,llama_free};
    common_chat_templates_ptr templates;
    std::unique_ptr<std::remove_pointer_t<ggml_backend_t>,decltype(&ggml_backend_free)> gpu{nullptr,ggml_backend_free};
    std::unique_ptr<HeadStore> heads;
    json metadata;
    bool router;
    int hidden=0,limit=0;
    static std::once_flag initialized;
public:
    explicit LanguageModel(const json &c):Model(c),router(c.at("role")=="router") {
        require(c.at("backend")=="llama.cpp","BACKEND_UNSUPPORTED");
        std::call_once(initialized,[]{llama_backend_init();ggml_backend_load_all();});
        auto desired=c.at("device").get<std::string>();require(desired=="metal" || desired=="cuda","GPU_REQUIRED");
        ggml_backend_dev_t device=nullptr;
        for(size_t i=0;i<ggml_backend_dev_count();++i) {
            auto d=ggml_backend_dev_get(i);auto type=ggml_backend_dev_type(d);
            std::string name=ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
            if((type==GGML_BACKEND_DEVICE_TYPE_GPU || type==GGML_BACKEND_DEVICE_TYPE_IGPU) &&
               ((desired=="metal" && name=="MTL") || (desired=="cuda" && name=="CUDA"))) {device=d;break;}
        }
        require(device,"GPU_BACKEND_UNAVAILABLE");
        ggml_backend_dev_t devices[]={device,nullptr};
        auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=-1;mp.no_host=true;
        // llama keeps token embeddings on CPU by default even with every layer
        // offloaded. The DeskPet contract includes the embedding lookup on GPU.
        llama_model_tensor_buft_override placement[]={{"token_embd.weight",ggml_backend_dev_buffer_type(device)},{nullptr,nullptr}};
        mp.tensor_buft_overrides=placement;
        weights.reset(llama_model_load_from_file(path("model").c_str(),mp));require(bool(weights),"GPU_MODEL_LOAD_FAILED");
        auto cp=llama_context_default_params();cp.n_ctx=c.value("contextTokens",2048);cp.n_batch=64;cp.n_ubatch=64;
        cp.n_seq_max=1;cp.embeddings=router;cp.pooling_type=LLAMA_POOLING_TYPE_NONE;cp.offload_kqv=true;cp.op_offload=true;
        context.reset(llama_init_from_model(weights.get(),cp));require(bool(context),"GPU_CONTEXT_FAILED");
        std::string chatTemplate;
        if(router) {
            std::ifstream in(path("chat_template"));require(bool(in),"CHAT_TEMPLATE_MISSING");
            chatTemplate.assign(std::istreambuf_iterator<char>(in),{});
        }
        templates=common_chat_templates_init(weights.get(),chatTemplate);require(bool(templates),"CHAT_TEMPLATE_MISSING");
        hidden=llama_model_n_embd(weights.get());limit=c.value("maxTokens",128);
        if(router) {
            metadata=read_json(path("config"));
            gpu.reset(ggml_backend_dev_init(device,nullptr));require(bool(gpu),"GPU_HEAD_BACKEND_FAILED");
            heads=std::make_unique<HeadStore>(path("heads"),gpu.get());
        }
    }
    ~LanguageModel(){stop();heads.reset();}
    std::string render(const json &messages) {
        common_chat_templates_inputs in;in.enable_thinking=false;in.add_generation_prompt=true;
        for(size_t i=0;i<messages.size();++i) {
            const auto &m=messages[i];common_chat_msg msg;msg.role=m.at("role");
            if(router) {
                require(msg.role=="user"||msg.role=="assistant","ROUTER_CONTEXT_ROLE_UNSUPPORTED");
            }
            msg.content=m.at("content");
            in.messages.push_back(std::move(msg));
        }
        return common_chat_templates_apply(templates.get(),in).prompt.str();
    }
    std::vector<llama_token> tokenize(const std::string &text) {
        auto vocab=llama_model_get_vocab(weights.get());
        int n=-llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,false,true);
        require(n>0 && n<=int(llama_n_ctx(context.get()))-limit,"INPUT_TOKEN_LIMIT");
        std::vector<llama_token> ids(n);require(llama_tokenize(vocab,text.data(),int(text.size()),ids.data(),n,false,true)==n,"TOKENIZE_ERROR");return ids;
    }
    std::string piece(llama_token token,bool special=false) {
        char buf[512];int n=llama_token_to_piece(llama_model_get_vocab(weights.get()),token,buf,sizeof(buf),0,special);
        require(n>=0,"TOKEN_PIECE_TOO_LARGE");return std::string(buf,size_t(n));
    }
    std::vector<float> evaluate(const std::vector<llama_token> &ids,Job &job,bool all) {
        llama_memory_clear(llama_get_memory(context.get()),true);
        std::vector<float> states;states.reserve(ids.size()*hidden);
        for(size_t first=0;first<ids.size();first+=64) {
            require(!job.cancelled,"CANCELLED");int count=int(std::min(size_t(64),ids.size()-first));auto b=llama_batch_init(count,0,1);
            for(int i=0;i<count;++i) {b.token[i]=ids[first+i];b.pos[i]=int(first)+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=all || first+i+1==ids.size();}b.n_tokens=count;
            int rc=llama_decode(context.get(),b);llama_batch_free(b);require(rc==0,"GPU_DECODE_FAILED");
            if(all) {auto p=llama_get_embeddings(context.get());require(p,"HIDDEN_STATE_UNAVAILABLE");states.insert(states.end(),p,p+size_t(count)*hidden);}
        }return states;
    }
    std::pair<std::string,double> generate_head(const std::string &head,const std::vector<llama_token> &prefix,Job &job,bool &finished) {
        auto hs=evaluate(prefix,job,true);std::vector<float> last(hs.end()-hidden,hs.end());
        std::string text;double logp=0;int count=0;finished=false;
        for(int i=0;i<limit;++i) {
            require(!job.cancelled,"CANCELLED");auto logits=heads->calculate(last,hidden,1,[&](auto c,auto x){return heads->linear(c,x,"generators."+head,false);});
            auto [token,p]=best(logits);logp+=std::log(std::max(p,1e-30));++count;
            if(token==llama_vocab_eos(llama_model_get_vocab(weights.get()))) {finished=true;break;}
            text+=piece(token);auto b=llama_batch_init(1,0,1);b.n_tokens=1;b.token[0]=token;b.pos[0]=int(prefix.size())+i;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
            auto rc=llama_decode(context.get(),b);llama_batch_free(b);require(rc==0,"GPU_HEAD_DECODE_FAILED");
            auto pstate=llama_get_embeddings_ith(context.get(),-1);require(pstate,"HIDDEN_STATE_UNAVAILABLE");last.assign(pstate,pstate+hidden);
        }return {text,count?std::exp(logp/count):0};
    }
    json infer(Job &job) override {
        auto messages=job.input.at("messages");
        auto contextMode=job.input.value("contextMode",std::string("legacy"));
        if(router) {
            require(contextMode=="legacy"||contextMode=="reference"||contextMode=="disabled","ROUTER_CONTEXT_MODE_UNSUPPORTED");
            require(!messages.empty()&&messages.back().at("role")=="user","ROUTER_CURRENT_MESSAGE_REQUIRED");
            auto current=job.input.value("currentMessageIndex",int(messages.size())-1);
            require(current==int(messages.size())-1,"ROUTER_CURRENT_MESSAGE_INDEX_INVALID");
            if(contextMode=="disabled")require(messages.size()==1,"ROUTER_CONTEXT_DISABLED_REQUIRES_CURRENT_ONLY");
        }
        auto prompt=render(messages);auto ids=tokenize(prompt);
        if(!router) {
            evaluate(ids,job,false);auto sampler=llama_sampler_init_greedy();std::string text;
            try {for(int i=0;i<limit;++i) {
                require(!job.cancelled,"CANCELLED");auto token=llama_sampler_sample(sampler,context.get(),-1);
                if(llama_vocab_is_eog(llama_model_get_vocab(weights.get()),token)) break;
                text+=piece(token);auto b=llama_batch_init(1,0,1);b.n_tokens=1;b.token[0]=token;b.pos[0]=int(ids.size())+i;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
                auto rc=llama_decode(context.get(),b);llama_batch_free(b);require(rc==0,"GPU_DECODE_FAILED");
            }}catch(...){llama_sampler_free(sampler);throw;}llama_sampler_free(sampler);
            require(text.find("<think>")==std::string::npos,"UNEXPECTED_REASONING_OUTPUT");return {{"text",text}};
        }
        require(ids.size()<=metadata.at("max_input_tokens").get<size_t>(),"ROUTER_INPUT_TOKEN_LIMIT");
        auto states=evaluate(ids,job,true);std::vector<float> pooled(states.end()-hidden,states.end());
        json values=json::object(),scores=json::object(),spans=json::object(),finished=json::object();
        const std::vector<std::string> classes={"Intention","Operation","Priority"}, spanHeads={"Target","Name","Checkpoint","DueAt"};
        for(const auto &name:classes) {
            auto logits=heads->calculate(pooled,hidden,1,[&](auto c,auto x){return heads->linear(c,x,"classifiers."+name);});
            auto [index,score]=best(logits);values[name]=metadata.at("labels").at(name).at(index);scores[name]=score;
        }
        // Match training's marker-based template mapping. A repeated sentence or
        // a role word inside a message must never select the template's text.
        struct Range{size_t a,b,left;int message;};std::vector<Range> ranges;
        auto marked=messages;std::vector<std::string> markers;
        for(size_t i=0;i<messages.size();++i) {
            auto marker="__deskpet_message_"+std::to_string(i)+"__";
            require(prompt.find(marker)==std::string::npos,"MESSAGE_MARKER_COLLISION");
            markers.push_back(marker);marked[i]["content"]=marker;
        }
        auto pattern=render(marked);std::string mapped;size_t cursor=0;
        for(size_t i=0;i<messages.size();++i) {
            auto at=pattern.find(markers[i],cursor);require(at!=std::string::npos,"MESSAGE_ALIGNMENT_FAILED");
            mapped+=pattern.substr(cursor,at-cursor);std::string raw=messages[i].at("content");auto [a,b]=trimmed(raw);
            size_t begin=mapped.size();mapped+=raw.substr(a,b-a);ranges.push_back({begin,mapped.size(),characters(raw.substr(0,a)),int(i)});
            cursor=at+markers[i].size();
        }
        mapped+=pattern.substr(cursor);require(mapped==prompt,"MESSAGE_ALIGNMENT_FAILED");
        std::vector<std::pair<size_t,size_t>> offsets;std::string restored;
        for(auto token:ids){size_t a=restored.size();restored+=piece(token,true);offsets.emplace_back(a,restored.size());}
        require(restored==prompt,"TOKEN_OFFSET_ALIGNMENT_FAILED");
        // Byte-fallback tokens share a Unicode scalar offset in the training
        // tokenizer. Expand byte fragments to those same scalar boundaries.
        for(auto &o:offsets) {
            while(o.first>0&&(uint8_t(prompt[o.first])&0xc0)==0x80)--o.first;
            while(o.second<prompt.size()&&(uint8_t(prompt[o.second])&0xc0)==0x80)++o.second;
        }
        int chars=metadata.at("max_token_chars").get<int>()+1;
        for(const auto &name:spanHeads) {
            values[name]=nullptr;scores[name]=0.0;if(values["Intention"]!="eureka")continue;
            auto compute=[&](const std::string &part){return heads->calculate(states,hidden,int(ids.size()),[&](auto c,auto x){
                auto last=ggml_view_2d(c,x,hidden,1,x->nb[1],(ids.size()-1)*x->nb[1]);
                auto repeated=ggml_repeat(c,last,x);auto features=ggml_concat(c,x,repeated,0);
                return heads->linear(c,features,"spans."+name+"."+part);
            });};
            auto positions=compute("positions"),boundaries=compute("boundaries");
            double max=-INFINITY,sum=0;int start=-1,end=-1,range=-1;
            for(size_t r=0;r<ranges.size();++r) for(size_t a=0;a<ids.size();++a) if(offsets[a].first>=ranges[r].a && offsets[a].second<=ranges[r].b && offsets[a].first<offsets[a].second)
                for(size_t b=a;b<ids.size() && offsets[b].second<=ranges[r].b;++b) {
                    double score=positions[2*a]+positions[2*b+1];if(score>max){sum=sum*std::exp(max-score)+1;max=score;start=int(a);end=int(b);range=int(r);}else sum+=std::exp(score-max);
                }
            if(start<0)continue;
            std::string s=prompt.substr(offsets[start].first,offsets[start].second-offsets[start].first),e=prompt.substr(offsets[end].first,offsets[end].second-offsets[end].first);
            auto sb=std::vector<float>(boundaries.begin()+start*chars*2,boundaries.begin()+start*chars*2+std::min(size_t(chars),characters(s)));
            auto eb=std::vector<float>(boundaries.begin()+end*chars*2+chars,boundaries.begin()+end*chars*2+chars+std::min(size_t(chars),characters(e)+1));if(sb.empty()||eb.size()<2)continue;eb[0]=-INFINITY;
            auto [si,sp]=best(sb);auto [ei,ep]=best(eb);
            size_t a=offsets[start].first+char_byte(s,si),b=offsets[end].first+char_byte(e,ei);
            if(a>=b) {
                double score=-INFINITY;int bs=-1,be=-1;
                for(size_t i=0;i<sb.size();++i)for(size_t j=i+1;j<eb.size();++j)
                    if(double(sb[i])+eb[j]>score){score=double(sb[i])+eb[j];bs=int(i);be=int(j);}
                if(bs<0)continue;
                auto probability=[](const std::vector<float> &v,int index){float m=*std::max_element(v.begin(),v.end());double sum=0;for(auto f:v)sum+=std::exp(double(f-m));return std::exp(double(v[index]-m))/sum;};
                si=bs;ei=be;sp=probability(sb,si);ep=probability(eb,ei);
                a=offsets[start].first+char_byte(s,si);b=offsets[end].first+char_byte(e,ei);
            }
            if(a>=b)continue;
            auto &r=ranges[range];auto text=prompt.substr(a,b-a);
            spans[name]={{"message",r.message},{"start",r.left+characters(prompt.substr(r.a,a-r.a))},{"end",r.left+characters(prompt.substr(r.a,b-r.a))},{"text",text}};
            values[name]=text;scores[name]=std::cbrt(std::max(1e-12,sp*ep/sum));
        }
        auto operation=values["Operation"].get<std::string>();bool isEureka=values["Intention"]=="eureka";
        bool task=isEureka && (operation=="create" || operation=="add_task"),note=isEureka && (operation=="create" || operation=="add_note" || operation=="update_item");
        for(const auto &name:std::vector<std::string>{"Task","Note"}) {values[name]=nullptr;scores[name]=0.0;
            if((name=="Task"&&task)||(name=="Note"&&note)){bool done;auto value=generate_head(name,ids,job,done);values[name]=value.first;scores[name]=value.second;finished[name]=done;require(done,"ROUTER_GENERATION_TRUNCATED");}}
        std::vector<float> features;
        for(const auto &h:std::vector<std::string>{"Intention","Operation","Priority","Target","Name","Checkpoint","DueAt","Task","Note"})
            features.push_back(h=="Intention" || (isEureka && (h!="Task"||task) && (h!="Note"||note)) ? scores[h].get<float>() : 0.0f);
        require(metadata.value("confidence_calibrated",false),"ROUTER_CONFIDENCE_UNAVAILABLE");
        auto conf=heads->calculate(features,9,1,[&](auto c,auto x){return ggml_sigmoid(c,heads->linear(c,ggml_gelu(c,heads->linear(c,x,"confidence.0")),"confidence.2"));});
        auto head=[&](const std::string &h){return json{{"value",isEureka?values[h]:json(nullptr)},{"confidence",isEureka?scores[h]:json(0.0)}};};
        std::string intent="unclear";
        if(values["Intention"]=="talk")intent="chat";else if(values["Intention"]=="llm")intent="llm";
        else if(isEureka)intent="eureka."+operation;
        return {{"intention",{{"value",intent},{"confidence",scores["Intention"]}}},{"modelIntention",{{"value",values["Intention"]},{"confidence",scores["Intention"]}}},
            {"operation",head("Operation")},{"target",head("Target")},{"name",head("Name")},{"checkpoint",head("Checkpoint")},
            {"dueAt",{{"value",nullptr},{"confidence",scores["DueAt"]}}},{"priority",head("Priority")},{"task",head("Task")},{"note",head("Note")},
            {"spans",spans},{"modelConfidence",conf.at(0)},{"generationFinished",finished}};
    }
};
std::once_flag LanguageModel::initialized;
}
std::shared_ptr<Model> make_language_model(const json &c) {return std::make_shared<LanguageModel>(c);}
