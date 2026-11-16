#include "runtime.hpp"
#include "whisper.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#ifdef DESKPET_SPEECH
#include "sherpa-onnx/c-api/c-api.h"
#include "onnxruntime_cxx_api.h"
#endif
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
#include <tokenizers_cpp.h>
#endif
#if defined(DESKPET_HAS_TENSORRT)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
namespace {
std::string trim_whitespace(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

class WhisperModel final : public Model {
    whisper_context *context=nullptr;
public:
    explicit WhisperModel(const json &c):Model(c) {
        require(c.at("device")=="cpu","STT_DEVICE_MUST_BE_CPU");
        auto p=whisper_context_default_params();p.use_gpu=false;
        context=whisper_init_from_file_with_params(path("model").c_str(),p);require(context,"STT_LOAD_FAILED");
    }
    ~WhisperModel(){stop();if(context)whisper_free(context);}
    json infer(Job &job) override {
        auto pcm=decode_pcm(job.input.at("pcm"));require(!pcm.empty()&&pcm.size()<=480000,"STT_AUDIO_LIMIT");
        auto p=whisper_full_default_params(WHISPER_SAMPLING_GREEDY);p.language="ko";p.translate=false;p.no_context=true;
        p.print_realtime=false;p.print_progress=false;p.print_timestamps=false;p.print_special=false;
        p.n_threads=config.value("threads",2);p.token_timestamps=true;
        p.abort_callback=[](void *data){return static_cast<Job *>(data)->cancelled.load();};p.abort_callback_user_data=&job;
        require(whisper_full(context,p,pcm.data(),int(pcm.size()))==0,"STT_INFERENCE_FAILED");
        std::string text;double logProbability=0;size_t count=0;
        for(int s=0;s<whisper_full_n_segments(context);++s) {
            text+=whisper_full_get_segment_text(context,s);
            for(int i=0;i<whisper_full_n_tokens(context,s);++i) {auto token=whisper_full_get_token_data(context,s,i);
                if(token.id<whisper_token_eot(context)){logProbability+=std::log(std::max(1e-12f,token.p));++count;}}
        }
        return {{"text",text},{"confidence",count?std::exp(logProbability/count):0.0}};
    }
};
#ifdef DESKPET_SPEECH
class Qwen3ASRModel final : public Model {
    const SherpaOnnxOfflineRecognizer *recognizer=nullptr;
public:
    explicit Qwen3ASRModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda","STT_DEVICE_UNSUPPORTED");
        std::string conv_frontend=path("conv_frontend");
        std::string encoder=path("encoder");
        std::string decoder=path("decoder");
        std::string tokenizer=path("tokenizer");
        std::string hotwords=c.value("hotwords",std::string());
        if(hotwords.empty()&&c.at("paths").contains("hotwords")&&c.at("paths").at("hotwords").is_string()) {
            hotwords=c.at("paths").at("hotwords").get<std::string>();
        }
        std::string provider=(dev=="cuda")?"cuda":"cpu";

        SherpaOnnxOfflineRecognizerConfig cfg{};
        std::memset(&cfg,0,sizeof(cfg));
        cfg.feat_config.sample_rate=16000;
        cfg.feat_config.feature_dim=128;
        cfg.decoding_method="greedy_search";
        cfg.model_config.num_threads=c.value("threads",2);
        cfg.model_config.debug=0;
        cfg.model_config.provider=provider.c_str();

        auto &q=cfg.model_config.qwen3_asr;
        q.conv_frontend=conv_frontend.c_str();
        q.encoder=encoder.c_str();
        q.decoder=decoder.c_str();
        q.tokenizer=tokenizer.c_str();
        int ctx=c.value("contextTokens",512);
        int max_new=c.value("maxTokens",128);
        q.max_total_len=ctx>0?ctx:512;
        q.max_new_tokens=max_new>0?max_new:128;
        q.temperature=1e-6f;
        q.top_p=0.8f;
        q.seed=42;
        q.hotwords=hotwords.empty()?nullptr:hotwords.c_str();

        recognizer=SherpaOnnxCreateOfflineRecognizer(&cfg);
        require(recognizer,"STT_LOAD_FAILED");
    }
    ~Qwen3ASRModel(){stop();if(recognizer)SherpaOnnxDestroyOfflineRecognizer(recognizer);}
    json infer(Job &job) override {
        auto pcm=decode_pcm(job.input.at("pcm"));
        require(!pcm.empty()&&pcm.size()<=480000,"STT_AUDIO_LIMIT");
        require(!job.cancelled.load(),"CANCELLED");
        struct StreamGuard {
            const SherpaOnnxOfflineStream *s=nullptr;
            const SherpaOnnxOfflineRecognizerResult *r=nullptr;
            ~StreamGuard() {
                if(r) SherpaOnnxDestroyOfflineRecognizerResult(r);
                if(s) SherpaOnnxDestroyOfflineStream(s);
            }
        } guard;
        guard.s=SherpaOnnxCreateOfflineStream(recognizer);
        require(guard.s,"STT_STREAM_FAILED");
        SherpaOnnxAcceptWaveformOffline(guard.s,16000,pcm.data(),static_cast<int32_t>(pcm.size()));
        require(!job.cancelled.load(),"CANCELLED");
        SherpaOnnxDecodeOfflineStream(recognizer,guard.s);
        require(!job.cancelled.load(),"CANCELLED");
        guard.r=SherpaOnnxGetOfflineStreamResult(guard.s);
        require(guard.r,"STT_INFERENCE_FAILED");
        std::string text=guard.r->text?trim_whitespace(guard.r->text):std::string();
        double confidence=text.empty()?0.0:1.0;
        if(guard.r->ys_log_probs&&guard.r->count>0) {
            double log_sum=0.0;
            for(int32_t i=0;i<guard.r->count;++i) log_sum+=static_cast<double>(guard.r->ys_log_probs[i]);
            confidence=std::exp(log_sum/static_cast<double>(guard.r->count));
        }
        return {{"text",text},{"confidence",confidence}};
    }
};

class VADModel final : public Model {
    const SherpaOnnxVoiceActivityDetector *vad=nullptr;
    bool speaking=false;
public:
    explicit VADModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda","VAD_DEVICE_UNSUPPORTED");
        std::string provider=(dev=="cuda")?"cuda":"cpu";
        auto file=path("model");SherpaOnnxVadModelConfig cfg{};cfg.silero_vad.model=file.c_str();
        cfg.silero_vad.threshold=0.5f;cfg.silero_vad.min_silence_duration=0.5f;cfg.silero_vad.min_speech_duration=0.1f;
        cfg.silero_vad.window_size=512;cfg.silero_vad.max_speech_duration=29;cfg.sample_rate=16000;
        cfg.num_threads=1;cfg.provider=provider.c_str();vad=SherpaOnnxCreateVoiceActivityDetector(&cfg,30);
        require(vad,"VAD_LOAD_FAILED");
    }
    ~VADModel(){stop();if(vad)SherpaOnnxDestroyVoiceActivityDetector(vad);}
    json infer(Job &job) override {
        if(job.input.value("reset",false)){SherpaOnnxVoiceActivityDetectorReset(vad);speaking=false;return {{"activity","silence"}};}
        auto pcm=decode_pcm(job.input.at("pcm"));SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad,pcm.data(),int(pcm.size()));
        bool now=SherpaOnnxVoiceActivityDetectorDetected(vad);std::string activity=now?"speech":speaking?"end":"silence";speaking=now;
        SherpaOnnxVoiceActivityDetectorClear(vad);return {{"activity",activity}};
    }
};
class TTSModel final : public Model {
    const SherpaOnnxOfflineTts *tts=nullptr;
public:
    explicit TTSModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda","TTS_DEVICE_UNSUPPORTED");
        std::string provider=(dev=="cuda")?"cuda":"cpu";
        auto p=config.at("paths");std::vector<std::string> files;
        for(auto key:{"duration_predictor","text_encoder","vector_estimator","vocoder","tts_json","unicode_indexer","voice_style"})files.push_back(p.at(key));
        SherpaOnnxOfflineTtsConfig cfg{};auto &s=cfg.model.supertonic;
        s.duration_predictor=files[0].c_str();s.text_encoder=files[1].c_str();s.vector_estimator=files[2].c_str();
        s.vocoder=files[3].c_str();s.tts_json=files[4].c_str();s.unicode_indexer=files[5].c_str();s.voice_style=files[6].c_str();
        cfg.model.num_threads=c.value("threads",2);cfg.model.provider=provider.c_str();cfg.max_num_sentences=1;cfg.silence_scale=1;
        tts=SherpaOnnxCreateOfflineTts(&cfg);require(tts,"TTS_LOAD_FAILED");
    }
    ~TTSModel(){stop();if(tts)SherpaOnnxDestroyOfflineTts(tts);}
    json infer(Job &job) override {
        auto text=job.input.at("text").get<std::string>();require(!text.empty() && text.size()<=8192,"TTS_TEXT_LIMIT");
        SherpaOnnxGenerationConfig cfg{};cfg.sid=0;cfg.speed=1;cfg.silence_scale=1;cfg.num_steps=8;cfg.extra="{\"lang\":\"ko\"}";
        auto audio=SherpaOnnxOfflineTtsGenerateWithConfig(tts,text.c_str(),&cfg,
            [](const float *,int32_t,float,void *arg)->int32_t{return !static_cast<Job *>(arg)->cancelled;},&job);
        require(audio,"TTS_INFERENCE_FAILED");
        try {auto encoded=encode_pcm(audio->samples,audio->n);int rate=audio->sample_rate;
            SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);return {{"pcm",encoded},{"sampleRate",rate}};
        }catch(...){SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);throw;}
    }
};
#if defined(DESKPET_HAS_TENSORRT) && defined(DESKPET_HAS_TOKENIZERS_CPP)
inline void speech_check_cuda(cudaError_t err,const char *what) {
    require(err==cudaSuccess,std::string(what)+": "+cudaGetErrorString(err));
}
class SpeechTrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity,const char *msg) noexcept override {
        if(severity<=Severity::kERROR) std::fprintf(stderr,"[TensorRT/TTS] %s\n",msg);
    }
} g_speech_trt_logger;

struct SpeechTrtDeleter {
    template <typename T> void operator()(T *obj) const noexcept { delete obj; }
};
template <typename T> using SpeechTrtPtr=std::unique_ptr<T,SpeechTrtDeleter>;

static std::string read_text_file(const std::string &path) {
    std::ifstream f(path,std::ios::binary);
    require(bool(f),"FILE_NOT_FOUND: "+path);
    return std::string(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
}
static bool speech_file_exists(const std::string &path) {
    if(path.empty()) return false;
    std::ifstream f(path,std::ios::binary);
    return f.good();
}
static std::string swap_suffix(const std::string &path,const std::string &from,const std::string &to) {
    if(path.size()>=from.size()&&path.compare(path.size()-from.size(),from.size(),from)==0) {
        return path.substr(0,path.size()-from.size())+to;
    }
    return path+to;
}
static std::string parent_dir(const std::string &path) {
    auto pos=path.find_last_of('/');
    return pos==std::string::npos?".":path.substr(0,pos);
}
static std::vector<int> extract_name_numbers(const std::string &s) {
    std::vector<int> nums;
    for(size_t i=0;i<s.size();) {
        if(std::isdigit(static_cast<unsigned char>(s[i]))) {
            int val=0;
            while(i<s.size()&&std::isdigit(static_cast<unsigned char>(s[i]))) {
                val=val*10+(s[i]-'0'); ++i;
            }
            nums.push_back(val);
        } else {
            ++i;
        }
    }
    return nums;
}
static bool natural_name_less(const std::string &a,const std::string &b) {
    auto na=extract_name_numbers(a), nb=extract_name_numbers(b);
    if(na!=nb) return na<nb;
    return a<b;
}
static size_t trt_dtype_bytes(nvinfer1::DataType dt) {
    switch(dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 4;
    }
}
static size_t dims_numel(const nvinfer1::Dims &d) {
    size_t n=1;
    for(int i=0;i<d.nbDims;++i) n*=static_cast<size_t>(std::max<int64_t>(0,d.d[i]));
    return n;
}

struct CudaDeviceBuffer {
    void *ptr=nullptr;
    size_t capacity=0;
    ~CudaDeviceBuffer(){reset();}
    CudaDeviceBuffer()=default;
    CudaDeviceBuffer(const CudaDeviceBuffer &)=delete;
    CudaDeviceBuffer &operator=(const CudaDeviceBuffer &)=delete;
    CudaDeviceBuffer(CudaDeviceBuffer &&o) noexcept:ptr(o.ptr),capacity(o.capacity){o.ptr=nullptr;o.capacity=0;}
    CudaDeviceBuffer &operator=(CudaDeviceBuffer &&o) noexcept {
        if(this!=&o){reset();ptr=o.ptr;capacity=o.capacity;o.ptr=nullptr;o.capacity=0;}
        return *this;
    }
    void ensure(size_t bytes) {
        if(bytes<=capacity&&ptr) return;
        reset();
        size_t alloc=std::max<size_t>(bytes,256);
        speech_check_cuda(cudaMalloc(&ptr,alloc),"cudaMalloc Qwen3TTS");
        capacity=alloc;
    }
    void reset(){if(ptr){cudaFree(ptr);ptr=nullptr;capacity=0;}}
};

struct TrtModule {
    SpeechTrtPtr<nvinfer1::ICudaEngine> engine;
    SpeechTrtPtr<nvinfer1::IExecutionContext> context;
    struct TensorInfo {
        std::string name;
        nvinfer1::TensorIOMode mode;
        nvinfer1::DataType dtype;
        nvinfer1::Dims shape;
        CudaDeviceBuffer buf;
    };
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string,size_t> by_name;
    std::vector<size_t> input_indices;
    std::vector<size_t> output_indices;

    void load_or_build(nvinfer1::IRuntime *runtime,const std::string &configured_path,
                       const std::function<void(nvinfer1::INetworkDefinition *,nvinfer1::IOptimizationProfile *)> &configure_profile,
                       bool allow_fp16) {
        std::string engine_path=configured_path;
        std::string onnx_path=swap_suffix(configured_path,".engine",".onnx");
        if(configured_path.size()>=5&&configured_path.compare(configured_path.size()-5,5,".onnx")==0) {
            onnx_path=configured_path;
            engine_path=swap_suffix(configured_path,".onnx",".engine");
        }
        if(speech_file_exists(engine_path)) {
            std::ifstream f(engine_path,std::ios::binary|std::ios::ate);
            require(bool(f),"TTS_ENGINE_READ_FAILED: "+engine_path);
            auto sz=f.tellg();require(sz>0,"TTS_ENGINE_EMPTY: "+engine_path);
            std::vector<char> bytes(static_cast<size_t>(sz));
            f.seekg(0,std::ios::beg);f.read(bytes.data(),sz);
            engine.reset(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
        } else if(speech_file_exists(onnx_path)) {
            SpeechTrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_speech_trt_logger));
            require(bool(builder),"TTS_TRT_BUILDER_FAILED");
            SpeechTrtPtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(
                1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
            SpeechTrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net,g_speech_trt_logger));
            require(parser->parseFromFile(onnx_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)),
                    "TTS_ONNX_PARSE_FAILED: "+onnx_path);
            SpeechTrtPtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
            if(allow_fp16) cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
            cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,512ULL<<20);
            auto *profile=builder->createOptimizationProfile();
            configure_profile(net.get(),profile);
            cfg->addOptimizationProfile(profile);
            SpeechTrtPtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net,*cfg));
            require(bool(plan),"TTS_TRT_BUILD_FAILED: "+onnx_path);
            std::string tmp=engine_path+".tmp";
            {
                std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
                if(out) out.write(reinterpret_cast<const char *>(plan->data()),static_cast<std::streamsize>(plan->size()));
            }
            std::rename(tmp.c_str(),engine_path.c_str());
            engine.reset(runtime->deserializeCudaEngine(plan->data(),plan->size()));
        } else {
            require(false,"TTS_MODEL_MISSING: "+configured_path);
        }
        require(bool(engine),"TTS_ENGINE_INIT_FAILED: "+engine_path);
        context.reset(engine->createExecutionContext());
        require(bool(context),"TTS_CONTEXT_INIT_FAILED: "+engine_path);

        int nb_io=engine->getNbIOTensors();
        tensors.resize(static_cast<size_t>(nb_io));
        for(int i=0;i<nb_io;++i) {
            const char *name=engine->getIOTensorName(i);
            auto &t=tensors[static_cast<size_t>(i)];
            t.name=name;
            t.mode=engine->getTensorIOMode(name);
            t.dtype=engine->getTensorDataType(name);
            t.shape=engine->getTensorShape(name);
            by_name[t.name]=static_cast<size_t>(i);
            if(t.mode==nvinfer1::TensorIOMode::kINPUT) input_indices.push_back(static_cast<size_t>(i));
            else output_indices.push_back(static_cast<size_t>(i));
        }
    }

    TensorInfo &require_tensor(const std::string &name) {
        auto it=by_name.find(name);
        require(it!=by_name.end(),"TTS_TRT_MISSING_TENSOR: "+name);
        return tensors[it->second];
    }
    void set_input_ints(const std::string &name,const nvinfer1::Dims &dims,const std::vector<int64_t> &vals,cudaStream_t stream) {
        auto &t=require_tensor(name);
        size_t n=dims_numel(dims);
        require(vals.size()==n,"TTS_INT_INPUT_SIZE_MISMATCH: "+name);
        require(context->setInputShape(t.name.c_str(),dims),"TTS_SET_INPUT_SHAPE_FAILED: "+name);
        t.buf.ensure(n*trt_dtype_bytes(t.dtype));
        if(n>0) {
            if(t.dtype==nvinfer1::DataType::kINT64) {
                speech_check_cuda(cudaMemcpyAsync(t.buf.ptr,vals.data(),n*sizeof(int64_t),cudaMemcpyHostToDevice,stream),"cudaMemcpyH2D int64");
            } else {
                require(t.dtype==nvinfer1::DataType::kINT32,"TTS_UNEXPECTED_INT_DTYPE: "+name);
                std::vector<int32_t> i32(n);
                for(size_t i=0;i<n;++i) i32[i]=static_cast<int32_t>(vals[i]);
                speech_check_cuda(cudaMemcpyAsync(t.buf.ptr,i32.data(),n*sizeof(int32_t),cudaMemcpyHostToDevice,stream),"cudaMemcpyH2D int32");
            }
        }
        require(context->setTensorAddress(t.name.c_str(),t.buf.ptr),"TTS_SET_IN_ADDR_FAILED: "+name);
    }
    void set_input_floats(const std::string &name,const nvinfer1::Dims &dims,const std::vector<float> &vals,cudaStream_t stream) {
        auto &t=require_tensor(name);
        size_t n=dims_numel(dims);
        require(vals.size()==n,"TTS_FLOAT_INPUT_SIZE_MISMATCH: "+name);
        require(context->setInputShape(t.name.c_str(),dims),"TTS_SET_INPUT_SHAPE_FAILED: "+name);
        t.buf.ensure(n*trt_dtype_bytes(t.dtype));
        if(n>0) {
            if(t.dtype==nvinfer1::DataType::kFLOAT) {
                speech_check_cuda(cudaMemcpyAsync(t.buf.ptr,vals.data(),n*sizeof(float),cudaMemcpyHostToDevice,stream),"cudaMemcpyH2D fp32");
            } else {
                require(t.dtype==nvinfer1::DataType::kHALF,"TTS_UNEXPECTED_FLOAT_DTYPE: "+name);
                std::vector<__half> f16(n);
                for(size_t i=0;i<n;++i) f16[i]=__float2half(vals[i]);
                speech_check_cuda(cudaMemcpyAsync(t.buf.ptr,f16.data(),n*sizeof(__half),cudaMemcpyHostToDevice,stream),"cudaMemcpyH2D fp16");
            }
        }
        require(context->setTensorAddress(t.name.c_str(),t.buf.ptr),"TTS_SET_IN_ADDR_FAILED: "+name);
    }
    void prepare_outputs_and_run(cudaStream_t stream) {
        for(size_t idx:output_indices) {
            auto &t=tensors[idx];
            auto dims=context->getTensorShape(t.name.c_str());
            for(int d=0;d<dims.nbDims;++d) require(dims.d[d]>=0,"TTS_OUTPUT_DIM_UNRESOLVED: "+t.name);
            t.buf.ensure(dims_numel(dims)*trt_dtype_bytes(t.dtype));
            require(context->setTensorAddress(t.name.c_str(),t.buf.ptr),"TTS_SET_OUT_ADDR_FAILED: "+t.name);
        }
        require(context->enqueueV3(stream),"TTS_TRT_ENQUEUE_FAILED");
    }
    std::vector<float> read_output_floats(const std::string &name,cudaStream_t stream,nvinfer1::Dims *out_dims=nullptr) {
        auto &t=require_tensor(name);
        auto dims=context->getTensorShape(t.name.c_str());
        if(out_dims) *out_dims=dims;
        size_t n=dims_numel(dims);
        std::vector<float> out(n);
        if(n==0) return out;
        if(t.dtype==nvinfer1::DataType::kFLOAT) {
            speech_check_cuda(cudaMemcpyAsync(out.data(),t.buf.ptr,n*sizeof(float),cudaMemcpyDeviceToHost,stream),"cudaMemcpyD2H fp32");
            speech_check_cuda(cudaStreamSynchronize(stream),"cudaStreamSync fp32");
        } else {
            require(t.dtype==nvinfer1::DataType::kHALF,"TTS_UNEXPECTED_OUT_DTYPE: "+name);
            std::vector<__half> f16(n);
            speech_check_cuda(cudaMemcpyAsync(f16.data(),t.buf.ptr,n*sizeof(__half),cudaMemcpyDeviceToHost,stream),"cudaMemcpyD2H fp16");
            speech_check_cuda(cudaStreamSynchronize(stream),"cudaStreamSync fp16");
            for(size_t i=0;i<n;++i) out[i]=__half2float(f16[i]);
        }
        return out;
    }
};

class Qwen3TTSTrtModel final : public Model {
    SpeechTrtPtr<nvinfer1::IRuntime> runtime;
    cudaStream_t stream=nullptr;
    TrtModule text_embed, codec_embed, residual_embed, talker, code_predictor, tok_decoder;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    json cfg;
    int64_t hidden_size=1024, vocab_size=3072, n_groups=16;
    int64_t im_start_id=151644, assistant_id=77091, newline_id=198;
    int64_t tts_bos_id=151672, tts_eos_id=151673, tts_pad_id=151671;
    int64_t codec_bos_id=2149, codec_eos_id=2150, codec_pad_id=2148;
    int64_t codec_think_id=2154, codec_nothink_id=2155, codec_think_bos_id=2156, codec_think_eos_id=2157;
    int64_t language_id=-1, speaker_id=-1;
    int max_new_tokens=2048;

    bool has_kv_cache=false;
    int kv_min_past=0;
    std::vector<std::string> past_names, present_names;
    std::vector<CudaDeviceBuffer> kv_past_bufs;

    static int sample_token(const float *logits,int64_t vocab,bool do_sample,int top_k,double top_p,double temperature,std::mt19937_64 &rng) {
        if(!do_sample||temperature<=0.0) {
            return static_cast<int>(std::max_element(logits,logits+vocab)-logits);
        }
        std::vector<double> scaled(static_cast<size_t>(vocab));
        double max_v=-std::numeric_limits<double>::infinity();
        const double inv_t=1.0/std::max(temperature,1e-6);
        for(int64_t i=0;i<vocab;++i) {
            double v=static_cast<double>(logits[i]);
            scaled[static_cast<size_t>(i)]=std::isfinite(v)?(v*inv_t):-std::numeric_limits<double>::infinity();
            if(scaled[static_cast<size_t>(i)]>max_v) max_v=scaled[static_cast<size_t>(i)];
        }
        if(!std::isfinite(max_v)) return 0;
        std::vector<std::pair<double,int>> items;
        items.reserve(static_cast<size_t>(vocab));
        for(int64_t i=0;i<vocab;++i) {
            if(std::isfinite(scaled[static_cast<size_t>(i)])) items.emplace_back(scaled[static_cast<size_t>(i)],static_cast<int>(i));
        }
        std::sort(items.begin(),items.end(),[](const auto &a,const auto &b){return a.first>b.first;});
        if(top_k>0&&static_cast<int>(items.size())>top_k) items.resize(static_cast<size_t>(top_k));
        double sum=0.0;
        for(auto &it:items){it.first=std::exp(it.first-max_v);sum+=it.first;}
        if(sum<=0.0||items.empty()) return items.empty()?0:items[0].second;
        for(auto &it:items) it.first/=sum;
        if(top_p>0.0&&top_p<1.0) {
            double csum=0.0;size_t keep=0;
            for(size_t i=0;i<items.size();++i){csum+=items[i].first;keep=i+1;if(csum>=top_p)break;}
            items.resize(keep);
            double kept_sum=0.0;for(const auto &it:items)kept_sum+=it.first;
            for(auto &it:items)it.first/=kept_sum;
        }
        std::uniform_real_distribution<double> u(0.0,1.0);
        double r=u(rng), acc=0.0;
        for(const auto &it:items){acc+=it.first;if(r<=acc)return it.second;}
        return items.back().second;
    }

    std::vector<float> embed_text(const std::vector<int64_t> &ids) {
        nvinfer1::Dims2 d{1,static_cast<int64_t>(ids.size())};
        text_embed.set_input_ints("text_ids",d,ids,stream);
        text_embed.prepare_outputs_and_run(stream);
        return text_embed.read_output_floats("text_embeds",stream);
    }
    std::vector<float> embed_codec(const std::vector<int64_t> &ids) {
        nvinfer1::Dims2 d{1,static_cast<int64_t>(ids.size())};
        codec_embed.set_input_ints("codec_ids",d,ids,stream);
        codec_embed.prepare_outputs_and_run(stream);
        return codec_embed.read_output_floats("codec_embeds",stream);
    }
    std::vector<float> run_residual_embed(const std::vector<int64_t> &codes16) {
        nvinfer1::Dims2 d{1,n_groups};
        residual_embed.set_input_ints("codec_ids",d,codes16,stream);
        residual_embed.prepare_outputs_and_run(stream);
        return residual_embed.read_output_floats("step_embed",stream);
    }
    std::vector<float> run_code_predictor(const std::vector<float> &last_hidden,const std::vector<int64_t> &codes16,nvinfer1::Dims &out_dims) {
        code_predictor.set_input_floats("talker_hidden",nvinfer1::Dims2{1,hidden_size},last_hidden,stream);
        code_predictor.set_input_ints("codec_ids",nvinfer1::Dims2{1,n_groups},codes16,stream);
        code_predictor.prepare_outputs_and_run(stream);
        return code_predictor.read_output_floats("group_logits",stream,&out_dims);
    }

public:
    explicit Qwen3TTSTrtModel(const json &c):Model(c) {
        require(c.at("device")=="cuda","TTS_TENSORRT_REQUIRES_CUDA");
        speech_check_cuda(cudaStreamCreate(&stream),"cudaStreamCreate Qwen3TTS");
        runtime.reset(nvinfer1::createInferRuntime(g_speech_trt_logger));
        require(bool(runtime),"TTS_TRT_RUNTIME_FAILED");

        cfg=read_json(path("config"));
        const auto &tc=cfg.at("talker_config");
        hidden_size=tc.at("hidden_size").get<int64_t>();
        vocab_size=tc.at("vocab_size").get<int64_t>();
        n_groups=tc.at("num_code_groups").get<int64_t>();
        im_start_id=cfg.at("im_start_token_id").get<int64_t>();
        assistant_id=cfg.at("assistant_token_id").get<int64_t>();
        tts_bos_id=cfg.at("tts_bos_token_id").get<int64_t>();
        tts_eos_id=cfg.at("tts_eos_token_id").get<int64_t>();
        tts_pad_id=cfg.at("tts_pad_token_id").get<int64_t>();
        codec_bos_id=tc.at("codec_bos_id").get<int64_t>();
        codec_eos_id=tc.at("codec_eos_token_id").get<int64_t>();
        codec_pad_id=tc.at("codec_pad_id").get<int64_t>();
        codec_think_id=tc.at("codec_think_id").get<int64_t>();
        codec_nothink_id=tc.at("codec_nothink_id").get<int64_t>();
        codec_think_bos_id=tc.at("codec_think_bos_id").get<int64_t>();
        codec_think_eos_id=tc.at("codec_think_eos_id").get<int64_t>();

        std::string lang=c.value("language",std::string("korean"));
        std::transform(lang.begin(),lang.end(),lang.begin(),[](unsigned char ch){return static_cast<char>(std::tolower(ch));});
        if(lang!="auto"&&tc.contains("codec_language_id")&&tc.at("codec_language_id").contains(lang)) {
            language_id=tc.at("codec_language_id").at(lang).get<int64_t>();
        }
        std::string spk=c.value("speaker",std::string("sohee"));
        std::transform(spk.begin(),spk.end(),spk.begin(),[](unsigned char ch){return static_cast<char>(std::tolower(ch));});
        if(!spk.empty()) {
            require(tc.contains("spk_id")&&tc.at("spk_id").contains(spk),"TTS_UNKNOWN_SPEAKER: "+spk);
            speaker_id=tc.at("spk_id").at(spk).get<int64_t>();
        }
        max_new_tokens=c.value("maxTokens",2048)>0?c.value("maxTokens",2048):2048;

        const auto &paths=c.at("paths");
        std::string tok_path=paths.contains("tokenizer")&&paths.at("tokenizer").is_string()?paths.at("tokenizer").get<std::string>():std::string();
        if(!tok_path.empty()&&speech_file_exists(tok_path)) {
            tokenizer=tokenizers::Tokenizer::FromBlobJSON(read_text_file(tok_path));
        } else {
            std::string dir=parent_dir(path("config"));
            std::string vocab_path=paths.contains("vocab")&&paths.at("vocab").is_string()?paths.at("vocab").get<std::string>():(dir+"/vocab.json");
            std::string merges_path=paths.contains("merges")&&paths.at("merges").is_string()?paths.at("merges").get<std::string>():(dir+"/merges.txt");
            tokenizer=tokenizers::Tokenizer::FromBlobByteLevelBPE(read_text_file(vocab_path),read_text_file(merges_path));
        }
        require(bool(tokenizer),"TTS_TOKENIZER_LOAD_FAILED");
        auto nl=tokenizer->Encode("\n");
        require(!nl.empty(),"TTS_NEWLINE_TOKEN_FAILED");
        newline_id=nl[0];

        auto seq2d_profile=[](nvinfer1::INetworkDefinition *net,nvinfer1::IOptimizationProfile *p) {
            for(int i=0;i<net->getNbInputs();++i) {
                auto *in=net->getInput(i);auto dims=in->getDimensions();
                if(dims.nbDims==2&&dims.d[1]<0) {
                    p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims2{1,1});
                    p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims2{1,32});
                    p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims2{1,512});
                }
            }
        };
        auto batch1_profile=[](nvinfer1::INetworkDefinition *net,nvinfer1::IOptimizationProfile *p) {
            for(int i=0;i<net->getNbInputs();++i) {
                auto *in=net->getInput(i);auto dims=in->getDimensions();
                nvinfer1::Dims fixed=dims;
                for(int d=0;d<fixed.nbDims;++d) if(fixed.d[d]<0) fixed.d[d]=1;
                p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kMIN,fixed);
                p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kOPT,fixed);
                p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kMAX,fixed);
            }
        };
        int64_t n_kv=tc.value("num_key_value_heads",8), head_dim=tc.value("head_dim",128), H=hidden_size;
        auto talker_profile=[H,n_kv,head_dim,this](nvinfer1::INetworkDefinition *net,nvinfer1::IOptimizationProfile *p) {
            int max_tot=std::min(4096,max_new_tokens+512);
            for(int i=0;i<net->getNbInputs();++i) {
                auto *in=net->getInput(i);const char *name=in->getName();auto dims=in->getDimensions();
                if(std::strcmp(name,"inputs_embeds")==0) {
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims3{1,1,H});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims3{1,1,H});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims3{1,512,H});
                } else if(std::strcmp(name,"position_ids")==0) {
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims3{3,1,1});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims3{3,1,1});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims3{3,1,512});
                } else if(std::strcmp(name,"attention_mask")==0) {
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims2{1,1});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims2{1,64});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims2{1,max_tot});
                } else if(dims.nbDims==4) {
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims4{1,n_kv,1,head_dim});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims4{1,n_kv,64,head_dim});
                    p->setDimensions(name,nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims4{1,n_kv,max_tot,head_dim});
                }
            }
        };
        auto decoder_profile=[](nvinfer1::INetworkDefinition *net,nvinfer1::IOptimizationProfile *p) {
            for(int i=0;i<net->getNbInputs();++i) {
                auto *in=net->getInput(i);
                nvinfer1::Dims3 fixed{1,25,16};
                p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kMIN,fixed);
                p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kOPT,fixed);
                p->setDimensions(in->getName(),nvinfer1::OptProfileSelector::kMAX,fixed);
            }
        };

        text_embed.load_or_build(runtime.get(),path("text_embed"),seq2d_profile,true);
        codec_embed.load_or_build(runtime.get(),path("codec_embed"),seq2d_profile,true);
        residual_embed.load_or_build(runtime.get(),path("residual_embed"),batch1_profile,true);
        talker.load_or_build(runtime.get(),path("talker"),talker_profile,true);
        code_predictor.load_or_build(runtime.get(),path("code_predictor"),batch1_profile,true);
        // tok_decoder stays FP32 (audio codec quantization degrades waveform quality).
        tok_decoder.load_or_build(runtime.get(),path("tok_decoder"),decoder_profile,false);

        for(size_t idx:talker.input_indices) {
            const auto &t=talker.tensors[idx];
            if(t.name!="inputs_embeds"&&t.name!="position_ids"&&t.name!="attention_mask") past_names.push_back(t.name);
        }
        for(size_t idx:talker.output_indices) {
            const auto &t=talker.tensors[idx];
            if(t.name!="logits"&&t.name!="hidden_states") present_names.push_back(t.name);
        }
        has_kv_cache=!past_names.empty();
        if(has_kv_cache) {
            require(past_names.size()==present_names.size(),"TTS_KV_TENSOR_MISMATCH");
            std::sort(past_names.begin(),past_names.end(),natural_name_less);
            std::sort(present_names.begin(),present_names.end(),natural_name_less);
            auto min_dims=talker.engine->getProfileShape(past_names[0].c_str(),0,nvinfer1::OptProfileSelector::kMIN);
            kv_min_past=static_cast<int>(std::max<int64_t>(0,min_dims.d[2]));
            kv_past_bufs.resize(past_names.size());
        }
    }
    ~Qwen3TTSTrtModel() override {
        stop();
        kv_past_bufs.clear();
        if(stream) cudaStreamDestroy(stream);
    }

    json infer(Job &job) override {
        auto text=job.input.at("text").get<std::string>();
        require(!text.empty()&&text.size()<=8192,"TTS_TEXT_LIMIT");
        require(!job.cancelled.load(),"CANCELLED");

        // 0) Tokenize text and fetch fixed special token embeddings.
        auto body_raw=tokenizer->Encode(text);
        require(!body_raw.empty(),"TTS_EMPTY_TOKENS");
        std::vector<int64_t> body_ids(body_raw.begin(),body_raw.end());

        auto spec=embed_text({tts_bos_id,tts_eos_id,tts_pad_id});
        std::vector<float> bos_e(spec.begin(),spec.begin()+hidden_size);
        std::vector<float> eos_e(spec.begin()+hidden_size,spec.begin()+2*hidden_size);
        std::vector<float> pad_e(spec.begin()+2*hidden_size,spec.begin()+3*hidden_size);
        auto role=embed_text({im_start_id,assistant_id,newline_id});

        // 1) Language + speaker + codec prefix embeddings.
        std::vector<int64_t> codec_prefill;
        if(language_id<0) codec_prefill={codec_nothink_id,codec_think_bos_id,codec_think_eos_id};
        else codec_prefill={codec_think_id,codec_think_bos_id,language_id,codec_think_eos_id};
        auto codec0=embed_codec(codec_prefill);
        auto codec1=embed_codec({codec_pad_id,codec_bos_id});
        std::vector<float> codec_input=std::move(codec0);
        if(speaker_id>=0) {
            auto spk_e=embed_codec({speaker_id});
            codec_input.insert(codec_input.end(),spk_e.begin(),spk_e.end());
        }
        codec_input.insert(codec_input.end(),codec1.begin(),codec1.end());
        int64_t m_codec=static_cast<int64_t>(codec_input.size()/hidden_size);

        // 2) Assemble talker prefill embeddings (role + codec_prefix + body + eos + codec_bos).
        std::vector<float> talker_in=std::move(role);
        for(int64_t i=0;i<m_codec-1;++i) {
            const auto &txt_vec=(i==m_codec-2)?bos_e:pad_e;
            for(int64_t h=0;h<hidden_size;++h) {
                talker_in.push_back(txt_vec[static_cast<size_t>(h)]+codec_input[static_cast<size_t>(i*hidden_size+h)]);
            }
        }
        std::vector<int64_t> pad_ids(body_ids.size()+1,codec_pad_id);
        auto text_body=embed_text(body_ids);
        auto codec_pads=embed_codec(pad_ids);
        for(size_t i=0;i<body_ids.size();++i) {
            for(int64_t h=0;h<hidden_size;++h) {
                talker_in.push_back(text_body[i*hidden_size+h]+codec_pads[i*hidden_size+h]);
            }
        }
        for(int64_t h=0;h<hidden_size;++h) {
            talker_in.push_back(eos_e[static_cast<size_t>(h)]+codec_pads[body_ids.size()*hidden_size+h]);
        }
        auto codec_bos_e=embed_codec({codec_bos_id});
        for(int64_t h=0;h<hidden_size;++h) {
            talker_in.push_back(pad_e[static_cast<size_t>(h)]+codec_bos_e[static_cast<size_t>(h)]);
        }

        // 3) Autoregressive loop with suppress + repetition penalty matching official generation config.
        std::mt19937_64 rng(0);
        const double temperature=0.9, top_p=1.0, repetition_penalty=1.05;
        const int top_k=50;
        std::vector<int> prev_first;
        std::vector<int64_t> all_codes;

        int64_t cur_seq=static_cast<int64_t>(talker_in.size()/hidden_size);
        int64_t total_pos=0;
        int64_t kv_past_len=0;
        std::vector<float> cur_embeds=std::move(talker_in);

        for(int step=0;step<max_new_tokens;++step) {
            require(!job.cancelled.load(),"CANCELLED");
            std::vector<int64_t> pos(static_cast<size_t>(3*cur_seq));
            for(int row=0;row<3;++row) {
                for(int64_t c=0;c<cur_seq;++c) pos[static_cast<size_t>(row*cur_seq+c)]=total_pos+c;
            }
            int64_t mask_prefix=(has_kv_cache&&step==0)?kv_min_past:0;
            int64_t kv_in_len=(step==0)?mask_prefix:kv_past_len;
            int64_t mask_len=kv_in_len+cur_seq;
            std::vector<int64_t> mask(static_cast<size_t>(mask_len),1);
            for(int64_t i=0;i<mask_prefix;++i) mask[static_cast<size_t>(i)]=0;

            talker.set_input_floats("inputs_embeds",nvinfer1::Dims3{1,cur_seq,hidden_size},cur_embeds,stream);
            talker.set_input_ints("position_ids",nvinfer1::Dims3{3,1,cur_seq},pos,stream);
            talker.set_input_ints("attention_mask",nvinfer1::Dims2{1,mask_len},mask,stream);

            if(has_kv_cache) {
                for(size_t k=0;k<past_names.size();++k) {
                    auto &pt=talker.require_tensor(past_names[k]);
                    nvinfer1::Dims kv_dims=pt.shape;
                    kv_dims.d[0]=1;
                    kv_dims.d[2]=kv_in_len;
                    require(talker.context->setInputShape(pt.name.c_str(),kv_dims),"TTS_SET_KV_SHAPE_FAILED");
                    size_t kv_bytes=dims_numel(kv_dims)*trt_dtype_bytes(pt.dtype);
                    kv_past_bufs[k].ensure(kv_bytes);
                    if(step==0&&kv_bytes>0) {
                        speech_check_cuda(cudaMemsetAsync(kv_past_bufs[k].ptr,0,kv_bytes,stream),"cudaMemset kv0");
                    }
                    require(talker.context->setTensorAddress(pt.name.c_str(),kv_past_bufs[k].ptr),"TTS_SET_KV_ADDR_FAILED");
                }
            }

            talker.prepare_outputs_and_run(stream);

            nvinfer1::Dims logits_dims{}, hidden_dims{};
            auto logits=talker.read_output_floats("logits",stream,&logits_dims);
            auto hidden=talker.read_output_floats("hidden_states",stream,&hidden_dims);

            if(has_kv_cache) {
                auto pres_dims=talker.context->getTensorShape(present_names[0].c_str());
                kv_past_len=pres_dims.d[2];
                for(size_t k=0;k<present_names.size();++k) {
                    const auto &pr=talker.require_tensor(present_names[k]);
                    size_t copy_bytes=dims_numel(pr.shape.nbDims?talker.context->getTensorShape(pr.name.c_str()):pres_dims)*trt_dtype_bytes(pr.dtype);
                    kv_past_bufs[k].ensure(copy_bytes);
                    if(copy_bytes>0) {
                        speech_check_cuda(cudaMemcpyAsync(kv_past_bufs[k].ptr,pr.buf.ptr,copy_bytes,cudaMemcpyDeviceToDevice,stream),"cudaMemcpyD2D kv");
                    }
                }
            }

            size_t last_off=static_cast<size_t>((cur_seq-1)*vocab_size);
            std::vector<float> first_logits(logits.begin()+last_off,logits.begin()+last_off+vocab_size);
            for(int64_t i=vocab_size-1024;i<vocab_size;++i) {
                if(i>=0&&i!=codec_eos_id) first_logits[static_cast<size_t>(i)]=-std::numeric_limits<float>::infinity();
            }
            for(int prev:prev_first) {
                if(prev>=0&&prev<vocab_size) {
                    float &v=first_logits[static_cast<size_t>(prev)];
                    if(std::isfinite(v)) v=(v<0.0f)?(v*static_cast<float>(repetition_penalty)):(v/static_cast<float>(repetition_penalty));
                }
            }
            int code0=sample_token(first_logits.data(),vocab_size,true,top_k,top_p,temperature,rng);
            if(code0==codec_eos_id) break;
            prev_first.push_back(code0);

            size_t hid_off=static_cast<size_t>((cur_seq-1)*hidden_size);
            std::vector<float> last_hidden(hidden.begin()+hid_off,hidden.begin()+hid_off+hidden_size);
            std::vector<int64_t> codes16(static_cast<size_t>(n_groups),0);
            codes16[0]=code0;
            for(int64_t j=1;j<n_groups;++j) {
                nvinfer1::Dims gdims{};
                auto gl=run_code_predictor(last_hidden,codes16,gdims);
                int64_t sub_vocab=gdims.d[2];
                const float *row_logits=gl.data()+static_cast<size_t>((j-1)*sub_vocab);
                codes16[static_cast<size_t>(j)]=sample_token(row_logits,sub_vocab,true,top_k,top_p,temperature,rng);
            }
            all_codes.insert(all_codes.end(),codes16.begin(),codes16.end());

            auto next_emb=run_residual_embed(codes16);
            for(int64_t h=0;h<hidden_size;++h) next_emb[static_cast<size_t>(h)]+=pad_e[static_cast<size_t>(h)];

            if(has_kv_cache) {
                total_pos+=cur_seq;
                cur_seq=1;
                cur_embeds=std::move(next_emb);
            } else {
                talker_in=std::move(cur_embeds);
                talker_in.insert(talker_in.end(),next_emb.begin(),next_emb.end());
                cur_embeds=std::move(talker_in);
                cur_seq+=1;
            }
        }

        int64_t total_frames=static_cast<int64_t>(all_codes.size()/n_groups);
        require(total_frames>0,"TTS_EMPTY_AUDIO");

        // 4) Decode discrete 12Hz frames via fixed 25-frame tok_decoder chunks -> 24kHz PCM.
        constexpr int64_t DEC_FRAMES=25;
        std::vector<float> pcm_out;
        for(int64_t s=0;s<total_frames;s+=DEC_FRAMES) {
            require(!job.cancelled.load(),"CANCELLED");
            int64_t rem=std::min(DEC_FRAMES,total_frames-s);
            std::vector<int64_t> chunk(static_cast<size_t>(DEC_FRAMES*n_groups));
            for(int64_t f=0;f<DEC_FRAMES;++f) {
                int64_t src_f=s+(f%rem);
                std::memcpy(&chunk[static_cast<size_t>(f*n_groups)],
                            &all_codes[static_cast<size_t>(src_f*n_groups)],
                            static_cast<size_t>(n_groups)*sizeof(int64_t));
            }
            tok_decoder.set_input_ints("audio_codes",nvinfer1::Dims3{1,DEC_FRAMES,n_groups},chunk,stream);
            tok_decoder.prepare_outputs_and_run(stream);
            auto wav=tok_decoder.read_output_floats("waveform",stream);
            size_t keep=(rem<DEC_FRAMES)
                ? static_cast<size_t>(std::max<int64_t>(0,std::llround(static_cast<double>(wav.size())*static_cast<double>(rem)/static_cast<double>(DEC_FRAMES))))
                : wav.size();
            keep=std::min(keep,wav.size());
            pcm_out.insert(pcm_out.end(),wav.begin(),wav.begin()+static_cast<ptrdiff_t>(keep));
        }
        require(!pcm_out.empty(),"TTS_EMPTY_WAVEFORM");
        return {{"pcm",encode_pcm(pcm_out.data(),pcm_out.size())},{"sampleRate",24000}};
    }
};
#endif
static std::vector<std::string> utf8parts(const std::string &s) {
    std::vector<std::string> out;for(size_t i=0;i<s.size();) {auto c=uint8_t(s[i]);size_t n=c<128?1:(c&0xe0)==0xc0?2:(c&0xf0)==0xe0?3:4;
        require(i+n<=s.size(),"INVALID_UTF8");out.push_back(s.substr(i,n));i+=n;}return out;
}
class EmotionModel final : public Model {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"deskpet"};Ort::Session session{nullptr};
    json vocab;std::vector<std::string> inputNames,outputNames;
public:
    explicit EmotionModel(const json &c):Model(c) {
        require(c.at("device")=="cpu","EMOTION_DEVICE_MUST_BE_CPU");
        Ort::SessionOptions options;options.SetIntraOpNumThreads(c.value("threads",2));
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session=Ort::Session(env,path("model").c_str(),options);
        auto tokenizer=read_json(path("tokenizer"));vocab=tokenizer.at("model").at("vocab");
        require(tokenizer.at("model").at("type")=="WordPiece","EMOTION_TOKENIZER_UNSUPPORTED");
        Ort::AllocatorWithDefaultOptions allocator;
        for(size_t i=0;i<session.GetInputCount();++i)inputNames.push_back(session.GetInputNameAllocated(i,allocator).get());
        for(size_t i=0;i<session.GetOutputCount();++i)outputNames.push_back(session.GetOutputNameAllocated(i,allocator).get());
    }
    ~EmotionModel(){stop();}
    json infer(Job &job) override {
        auto words=job.input.at("words").get<std::vector<std::string>>();
        std::vector<int64_t> ids{vocab.at("[CLS]")};
        for(const auto &w:words) {
            auto parts=utf8parts(w);std::vector<int64_t> tokens;size_t pos=0;
            if(parts.size()>100){tokens={vocab.at("[UNK]")};pos=parts.size();}
            while(pos<parts.size()) {bool found=false;
                for(size_t end=parts.size();end>pos;--end){std::string candidate=pos?"##":"";for(size_t j=pos;j<end;++j)candidate+=parts[j];
                    if(vocab.contains(candidate)){tokens.push_back(vocab.at(candidate));pos=end;found=true;break;}}
                if(!found){tokens={vocab.at("[UNK]")};break;}
            }for(auto id:tokens){if(ids.size()==127)break;ids.push_back(id);}if(ids.size()==127)break;
        }ids.push_back(vocab.at("[SEP]"));
        std::vector<int64_t> mask(ids.size(),1);ids.resize(128,vocab.at("[PAD]").get<int64_t>());mask.resize(128,0);
        std::vector<int64_t> types(ids.size(),0);int64_t shape[]={1,int64_t(ids.size())};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::vector<Ort::Value> inputs;std::vector<const char *> names,outputs;
        for(auto &name:inputNames){names.push_back(name.c_str());auto &v=name=="input_ids"?ids:name=="attention_mask"?mask:types;
            require(name=="input_ids" || name=="attention_mask" || name=="token_type_ids","EMOTION_INPUT_UNSUPPORTED");
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory,v.data(),v.size(),shape,2));}
        for(auto &n:outputNames)outputs.push_back(n.c_str());auto result=session.Run(Ort::RunOptions{nullptr},names.data(),inputs.data(),inputs.size(),outputs.data(),outputs.size());
        require(result[0].GetTensorTypeAndShapeInfo().GetElementCount()==2,"EMOTION_LABEL_MISMATCH");auto logits=result[0].GetTensorData<float>();
        return {{"positive",1.0/(1.0+std::exp(double(logits[0]-logits[1])))}};
    }
};
#endif
}
std::shared_ptr<Model> make_speech_model(const json &c) {
    std::string role=c.at("role"),backend=c.at("backend"),device=c.at("device");
    require(device=="cpu"||device=="cuda","SPEECH_DEVICE_UNSUPPORTED");
    if(role=="stt"&&backend=="whisper.cpp") return std::make_shared<WhisperModel>(c);
#ifdef DESKPET_SPEECH
    if(role=="stt"&&backend=="sherpa-onnx") return std::make_shared<Qwen3ASRModel>(c);
    if(role=="vad"&&backend=="sherpa-onnx") return std::make_shared<VADModel>(c);
    if(role=="tts"&&backend=="sherpa-onnx") return std::make_shared<TTSModel>(c);
#if defined(DESKPET_HAS_TENSORRT) && defined(DESKPET_HAS_TOKENIZERS_CPP)
    if(role=="tts"&&backend=="tensorrt") return std::make_shared<Qwen3TTSTrtModel>(c);
#endif
    if(role=="emotion"&&backend=="onnxruntime") return std::make_shared<EmotionModel>(c);
#endif
    if(role=="tts"&&backend=="tensorrt") throw std::runtime_error("TENSORRT_TTS_NOT_AVAILABLE");
    require((role=="stt"&&(backend=="sherpa-onnx"||backend=="whisper.cpp")) ||
            ((role=="vad"||role=="tts")&&backend=="sherpa-onnx") ||
            (role=="tts"&&backend=="tensorrt") ||
            (role=="emotion"&&backend=="onnxruntime"),"BACKEND_UNSUPPORTED");
    throw std::runtime_error("NATIVE_DEPENDENCY_MISSING: "+role);
}
