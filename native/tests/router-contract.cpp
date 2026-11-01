#include "../router-head-graph.hpp"
#include "chat.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char **argv) {
    try {
        auto ctx=ggml_init({2*1024*1024,nullptr,true});check(ctx,"graph context");
        // Cover classifiers, the nine-feature Confidence MLP, and Span heads.
        // Only graph descriptors are constructed: no weights, CPU inference,
        // Metal device or GPU operation is loaded/executed by this test.
        for(auto dimensions:std::vector<std::pair<int,int>>{{1024,4},{9,32},{32,1},{2048,258}}) {
            auto [inputs,outputs]=dimensions;
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,inputs,1);
            auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,inputs,outputs);
            auto b=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,outputs);
            auto y=deskpet_head_linear(ctx,x,w,b);
            check(y->op==GGML_OP_ADD,"missing bias add");
            check(y->src[0]->type==y->src[1]->type,"mixed-type ADD would be rejected by Metal");
            check(y->type==GGML_TYPE_F32&&y->ne[0]==outputs,"linear output shape/type");
            check(w->type==GGML_TYPE_F16&&b->type==GGML_TYPE_F16,"stored parameter precision changed");
            check(y->src[1]->src[0]==b,"bias cast must remain in the computation graph");
            auto noBias=deskpet_head_linear(ctx,x,w,nullptr);
            check(noBias->op==GGML_OP_MUL_MAT,"generator must not acquire a bias");
        }
        ggml_free(ctx);std::cout<<"FP16 parameter / FP32 activation graph contracts passed\n";
        if(argc==2) {
            std::ifstream file(argv[1]);check(bool(file),"missing training template");
            std::string source((std::istreambuf_iterator<char>(file)),{});
            auto templates=common_chat_templates_init(nullptr,source);
            common_chat_templates_inputs in;in.enable_thinking=false;in.add_generation_prompt=true;
            common_chat_msg user;user.role="user";user.content="안녕하세요";in.messages={user};
            auto prompt=common_chat_templates_apply(templates.get(),in).prompt.str();
            check(prompt=="<|im_start|>user\n안녕하세요<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","training prompt mismatch");
            std::cout<<"Korean training chat template contract passed\n";
        }
        return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
