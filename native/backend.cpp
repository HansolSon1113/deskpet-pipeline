#include "runtime.hpp"
#include <cstdlib>
#include <cstring>
namespace {
struct ModelHandle { std::shared_ptr<Model> model; };
struct JobHandle { std::shared_ptr<Model> model; std::shared_ptr<Job> job; };
void *create(const char *text,char **error) {
    try {
        auto c=json::parse(text); auto role=c.at("role").get<std::string>();
        auto m=(role=="router" || role=="output") ? make_language_model(c) : make_speech_model(c);
        m->start(); return new ModelHandle{std::move(m)};
    } catch(const std::exception &e) { *error=strdup(e.what()); return nullptr; }
}
void destroy(void *handle) { delete static_cast<ModelHandle *>(handle); }
void *submit(void *handle,const char *input,char **error) {
    try { auto m=static_cast<ModelHandle *>(handle)->model; return new JobHandle{m,m->submit(json::parse(input))}; }
    catch(const std::exception &e) { *error=strdup(e.what()); return nullptr; }
}
int poll(void *handle,const char **result,const char **error) {
    auto &j=*static_cast<JobHandle *>(handle)->job; int done=j.done.load(std::memory_order_acquire);
    if(done==1) *result=j.output.c_str(); if(done==-1) *error=j.error.c_str(); return done;
}
void cancel(void *handle) { static_cast<JobHandle *>(handle)->job->cancelled=true; }
void release(void *handle) { delete static_cast<JobHandle *>(handle); }
const dp_backend_api api{1,create,destroy,submit,poll,cancel,release,[](char *p){free(p);}};
}
extern "C" const dp_backend_api *deskpet_backend_v1() { return &api; }

std::vector<float> decode_pcm(const std::string &s) {
    const std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    require(s.size()<=2'000'000,"AUDIO_TOO_LARGE");
    std::vector<unsigned char> bytes; uint32_t bits=0; int n=0;
    for(char c:s) { if(c=='=') break; auto i=alphabet.find(c); require(i!=std::string::npos,"INVALID_PCM_BASE64");
        bits=(bits<<6)|uint32_t(i); n+=6; if(n>=8) {n-=8; bytes.push_back(uint8_t(bits>>n));} }
    require(bytes.size()%2==0,"INVALID_PCM_SIZE"); std::vector<float> pcm(bytes.size()/2);
    for(size_t i=0;i<pcm.size();++i) pcm[i]=int16_t(uint16_t(bytes[2*i])|(uint16_t(bytes[2*i+1])<<8))/32768.0f;
    return pcm;
}
std::string encode_pcm(const float *samples,size_t count) {
    require(count<=16'000'000,"TTS_TOO_LARGE");
    const char *abc="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> bytes(count*2); for(size_t i=0;i<count;++i) {
        float f=std::max(-1.0f,std::min(0.999969f,samples[i])); int16_t x=int16_t(f*32768.0f);
        bytes[2*i]=uint8_t(x); bytes[2*i+1]=uint8_t(uint16_t(x)>>8);
    }
    std::string out; out.reserve((bytes.size()+2)/3*4);
    for(size_t i=0;i<bytes.size();i+=3) { uint32_t v=uint32_t(bytes[i])<<16;
        if(i+1<bytes.size()) v|=uint32_t(bytes[i+1])<<8; if(i+2<bytes.size()) v|=bytes[i+2];
        out+=abc[(v>>18)&63];out+=abc[(v>>12)&63];out+=i+1<bytes.size()?abc[(v>>6)&63]:'=';out+=i+2<bytes.size()?abc[v&63]:'=';
    } return out;
}
