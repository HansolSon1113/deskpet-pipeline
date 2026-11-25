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
namespace {
std::string trim_whitespace(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}
std::string join_tokens(const char *const *tokens_arr, int32_t count, const char *fallback) {
    if (tokens_arr && count > 0) {
        std::string joined;
        for (int32_t i = 0; i < count; ++i) {
            if (!tokens_arr[i]) continue;
            std::string tok(tokens_arr[i]);
            for (size_t pos = 0; (pos = tok.find("\xe2\x96\x81", pos)) != std::string::npos;) {
                tok.replace(pos, 3, " ");
                pos += 1;
            }
            joined += tok;
        }
        std::string trimmed = trim_whitespace(joined);
        if (!trimmed.empty()) return trimmed;
    }
    return fallback ? trim_whitespace(fallback) : std::string();
}

class WhisperModel final : public Model {
    whisper_context *context=nullptr;
public:
    explicit WhisperModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","STT_DEVICE_UNSUPPORTED");
        auto p=whisper_context_default_params();p.use_gpu=(dev!="cpu");
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
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","STT_DEVICE_UNSUPPORTED");
        std::string conv_frontend=path("conv_frontend");
        std::string encoder=path("encoder");
        std::string decoder=path("decoder");
        std::string tokenizer=path("tokenizer");
        std::string hotwords=c.value("hotwords",std::string());
        if(hotwords.empty()&&c.at("paths").contains("hotwords")&&c.at("paths").at("hotwords").is_string()) {
            hotwords=c.at("paths").at("hotwords").get<std::string>();
        }
        std::string provider=(dev=="cuda")?"cuda":(dev=="metal"||dev=="coreml")?"coreml":"cpu";

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

class ZipformerModel final : public Model {
    const SherpaOnnxOfflineRecognizer *offline_rec=nullptr;
    const SherpaOnnxOnlineRecognizer *online_rec=nullptr;
    std::unordered_map<std::string, const SherpaOnnxOnlineStream *> streams;
    bool streaming=false;
public:
    explicit ZipformerModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","STT_DEVICE_UNSUPPORTED");
        std::string encoder=path("encoder");
        std::string decoder=path("decoder");
        std::string joiner=path("joiner");
        std::string tokens=path("tokens");
        std::string provider=(dev=="cuda")?"cuda":(dev=="metal"||dev=="coreml")?"coreml":"cpu";
        streaming=c.value("streaming",false)||(encoder.find("chunk-")!=std::string::npos);

        if(streaming) {
            SherpaOnnxOnlineRecognizerConfig cfg{};
            std::memset(&cfg,0,sizeof(cfg));
            cfg.feat_config.sample_rate=16000;
            cfg.feat_config.feature_dim=80;
            cfg.decoding_method="greedy_search";
            cfg.model_config.num_threads=c.value("threads",4);
            cfg.model_config.debug=0;
            cfg.model_config.provider=provider.c_str();
            cfg.model_config.tokens=tokens.c_str();
            cfg.model_config.transducer.encoder=encoder.c_str();
            cfg.model_config.transducer.decoder=decoder.c_str();
            cfg.model_config.transducer.joiner=joiner.c_str();
            online_rec=SherpaOnnxCreateOnlineRecognizer(&cfg);
            require(online_rec,"STT_LOAD_FAILED");
        } else {
            SherpaOnnxOfflineRecognizerConfig cfg{};
            std::memset(&cfg,0,sizeof(cfg));
            cfg.feat_config.sample_rate=16000;
            cfg.feat_config.feature_dim=80;
            cfg.decoding_method="greedy_search";
            cfg.model_config.num_threads=c.value("threads",4);
            cfg.model_config.debug=0;
            cfg.model_config.provider=provider.c_str();
            cfg.model_config.tokens=tokens.c_str();
            cfg.model_config.transducer.encoder=encoder.c_str();
            cfg.model_config.transducer.decoder=decoder.c_str();
            cfg.model_config.transducer.joiner=joiner.c_str();
            offline_rec=SherpaOnnxCreateOfflineRecognizer(&cfg);
            require(offline_rec,"STT_LOAD_FAILED");
        }
    }
    ~ZipformerModel() {
        stop();
        for (auto &kv : streams) {
            if (kv.second) SherpaOnnxDestroyOnlineStream(kv.second);
        }
        streams.clear();
        if(online_rec) SherpaOnnxDestroyOnlineRecognizer(online_rec);
        if(offline_rec) SherpaOnnxDestroyOfflineRecognizer(offline_rec);
    }
    json infer(Job &job) override {
        require(!job.cancelled.load(),"CANCELLED");
        if(streaming && job.input.contains("sessionId") && job.input["sessionId"].is_string()) {
            std::string sid=job.input.at("sessionId").get<std::string>();
            std::string action=job.input.value("action",std::string("end"));
            if(action=="cancel") {
                auto it=streams.find(sid);
                if(it!=streams.end()) {
                    if(it->second) SherpaOnnxDestroyOnlineStream(it->second);
                    streams.erase(it);
                }
                return {{"text",""},{"confidence",0.0}};
            }
            const SherpaOnnxOnlineStream *stream=nullptr;
            auto it=streams.find(sid);
            if(it!=streams.end()) {
                stream=it->second;
            } else {
                stream=SherpaOnnxCreateOnlineStream(online_rec);
                require(stream,"STT_STREAM_FAILED");
                streams[sid]=stream;
            }
            std::string b64=job.input.value("pcm",std::string());
            if(!b64.empty()) {
                auto pcm=decode_pcm(b64);
                require(pcm.size()<=480000,"STT_AUDIO_LIMIT");
                if(!pcm.empty()) {
                    SherpaOnnxOnlineStreamAcceptWaveform(stream,16000,pcm.data(),static_cast<int32_t>(pcm.size()));
                }
            }
            if(action=="end") {
                std::vector<float> tail(12800,0.0f); // 0.8s tail flush so the final chunk-64 flushes completely
                SherpaOnnxOnlineStreamAcceptWaveform(stream,16000,tail.data(),static_cast<int32_t>(tail.size()));
                SherpaOnnxOnlineStreamInputFinished(stream);
            }
            while(SherpaOnnxIsOnlineStreamReady(online_rec,stream)) {
                require(!job.cancelled.load(),"CANCELLED");
                SherpaOnnxDecodeOnlineStream(online_rec,stream);
            }
            const SherpaOnnxOnlineRecognizerResult *r=SherpaOnnxGetOnlineStreamResult(online_rec,stream);
            std::string text;
            if(r) {
                text=join_tokens(r->tokens_arr,r->count,r->text);
                SherpaOnnxDestroyOnlineRecognizerResult(r);
            }
            if(action=="end") {
                SherpaOnnxDestroyOnlineStream(stream);
                streams.erase(sid);
            }
            double confidence=text.empty()?0.0:1.0;
            return {{"text",text},{"confidence",confidence}};
        }
        auto pcm=decode_pcm(job.input.at("pcm"));
        require(!pcm.empty()&&pcm.size()<=480000,"STT_AUDIO_LIMIT");
        require(!job.cancelled.load(),"CANCELLED");
        if(streaming) {
            struct OnlineGuard {
                const SherpaOnnxOnlineStream *s=nullptr;
                const SherpaOnnxOnlineRecognizerResult *r=nullptr;
                ~OnlineGuard() {
                    if(r) SherpaOnnxDestroyOnlineRecognizerResult(r);
                    if(s) SherpaOnnxDestroyOnlineStream(s);
                }
            } guard;
            guard.s=SherpaOnnxCreateOnlineStream(online_rec);
            require(guard.s,"STT_STREAM_FAILED");
            SherpaOnnxOnlineStreamAcceptWaveform(guard.s,16000,pcm.data(),static_cast<int32_t>(pcm.size()));
            // Pad 0.8s tail silence so the final causal chunk (e.g. chunk-64) flushes completely.
            std::vector<float> tail(12800,0.0f);
            SherpaOnnxOnlineStreamAcceptWaveform(guard.s,16000,tail.data(),static_cast<int32_t>(tail.size()));
            SherpaOnnxOnlineStreamInputFinished(guard.s);
            while(SherpaOnnxIsOnlineStreamReady(online_rec,guard.s)) {
                require(!job.cancelled.load(),"CANCELLED");
                SherpaOnnxDecodeOnlineStream(online_rec,guard.s);
            }
            guard.r=SherpaOnnxGetOnlineStreamResult(online_rec,guard.s);
            require(guard.r,"STT_INFERENCE_FAILED");
            std::string text=join_tokens(guard.r->tokens_arr,guard.r->count,guard.r->text);
            double confidence=text.empty()?0.0:1.0;
            return {{"text",text},{"confidence",confidence}};
        } else {
            struct OfflineGuard {
                const SherpaOnnxOfflineStream *s=nullptr;
                const SherpaOnnxOfflineRecognizerResult *r=nullptr;
                ~OfflineGuard() {
                    if(r) SherpaOnnxDestroyOfflineRecognizerResult(r);
                    if(s) SherpaOnnxDestroyOfflineStream(s);
                }
            } guard;
            guard.s=SherpaOnnxCreateOfflineStream(offline_rec);
            require(guard.s,"STT_STREAM_FAILED");
            SherpaOnnxAcceptWaveformOffline(guard.s,16000,pcm.data(),static_cast<int32_t>(pcm.size()));
            require(!job.cancelled.load(),"CANCELLED");
            SherpaOnnxDecodeOfflineStream(offline_rec,guard.s);
            require(!job.cancelled.load(),"CANCELLED");
            guard.r=SherpaOnnxGetOfflineStreamResult(guard.s);
            require(guard.r,"STT_INFERENCE_FAILED");
            std::string text=join_tokens(guard.r->tokens_arr,guard.r->count,guard.r->text);
            double confidence=text.empty()?0.0:1.0;
            if(guard.r->ys_log_probs&&guard.r->count>0) {
                double log_sum=0.0;
                for(int32_t i=0;i<guard.r->count;++i) log_sum+=static_cast<double>(guard.r->ys_log_probs[i]);
                confidence=std::exp(log_sum/static_cast<double>(guard.r->count));
            }
            return {{"text",text},{"confidence",confidence}};
        }
    }
};

class VADModel final : public Model {
    const SherpaOnnxVoiceActivityDetector *vad=nullptr;
    bool speaking=false;
public:
    explicit VADModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","VAD_DEVICE_UNSUPPORTED");
        std::string provider=(dev=="cuda")?"cuda":(dev=="metal"||dev=="coreml")?"coreml":"cpu";
        auto file=path("model");SherpaOnnxVadModelConfig cfg{};cfg.silero_vad.model=file.c_str();
        cfg.silero_vad.threshold=c.value("threshold",0.5f);cfg.silero_vad.min_silence_duration=c.value("minSilenceDuration",0.3f);cfg.silero_vad.min_speech_duration=c.value("minSpeechDuration",0.08f);
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
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","TTS_DEVICE_UNSUPPORTED");
        std::string provider=(dev=="cuda")?"cuda":(dev=="metal"||dev=="coreml")?"coreml":"cpu";
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

struct OrtModule {
    Ort::Session session{nullptr};
    std::vector<std::string> in_names, out_names;
    std::vector<const char *> in_cstrs, out_cstrs;
    std::unordered_map<std::string,size_t> in_idx, out_idx;
    std::vector<ONNXTensorElementDataType> in_dtypes, out_dtypes;
    std::vector<std::vector<int64_t>> in_shapes, out_shapes;

    static Ort::Session create_session(Ort::Env &env,const std::string &onnx_path,const std::string &device,int threads) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(__APPLE__)
        options.DisableMemPattern();
        options.AddConfigEntry("session.memory.enable_memory_arena_shrinkage", "cpu:0");
#endif
        if(device=="cuda") {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.cudnn_conv_algo_search=OrtCudnnConvAlgoSearchHeuristic;
            options.AppendExecutionProvider_CUDA(cuda_opts);
        }
#if defined(__APPLE__)
        else if(device=="metal"||device=="coreml") {
            options.AppendExecutionProvider("CoreML",{
                {"ModelFormat","MLProgram"},
                {"MLComputeUnits","CPUAndGPU"},
                {"AllowLowPrecisionAccumulationOnGPU","1"}
            });
        }
#endif
        return Ort::Session(env,onnx_path.c_str(),options);
    }

    void load(Ort::Env &env,const std::string &configured_path,const std::string &device,int threads) {
        std::string onnx_path=configured_path;
        if(configured_path.size()>=7&&configured_path.compare(configured_path.size()-7,7,".engine")==0) {
            std::string cand=swap_suffix(configured_path,".engine",".onnx");
            if(speech_file_exists(cand)) onnx_path=cand;
        }
        require(speech_file_exists(onnx_path),"TTS_MODEL_MISSING: "+configured_path);

        session=create_session(env,onnx_path,device,threads);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t n_in=session.GetInputCount();
        in_names.reserve(n_in); in_cstrs.reserve(n_in); in_dtypes.reserve(n_in); in_shapes.reserve(n_in);
        for(size_t i=0;i<n_in;++i) {
            in_names.emplace_back(session.GetInputNameAllocated(i,allocator).get());
            in_idx[in_names.back()]=i;
            auto ti=session.GetInputTypeInfo(i);
            auto info=ti.GetTensorTypeAndShapeInfo();
            in_dtypes.push_back(info.GetElementType());
            in_shapes.push_back(info.GetShape());
        }
        for(const auto &name:in_names) in_cstrs.push_back(name.c_str());

        size_t n_out=session.GetOutputCount();
        out_names.reserve(n_out); out_cstrs.reserve(n_out); out_dtypes.reserve(n_out); out_shapes.reserve(n_out);
        for(size_t i=0;i<n_out;++i) {
            out_names.emplace_back(session.GetOutputNameAllocated(i,allocator).get());
            out_idx[out_names.back()]=i;
            auto to=session.GetOutputTypeInfo(i);
            auto info=to.GetTensorTypeAndShapeInfo();
            out_dtypes.push_back(info.GetElementType());
            out_shapes.push_back(info.GetShape());
        }
        for(const auto &name:out_names) out_cstrs.push_back(name.c_str());
    }

    size_t require_input(const std::string &name) const {
        auto it=in_idx.find(name);
        require(it!=in_idx.end(),"TTS_ORT_MISSING_INPUT: "+name);
        return it->second;
    }
    size_t require_output(const std::string &name) const {
        auto it=out_idx.find(name);
        require(it!=out_idx.end(),"TTS_ORT_MISSING_OUTPUT: "+name);
        return it->second;
    }

    static Ort::Value make_int_tensor(const OrtMemoryInfo *mem,ONNXTensorElementDataType dt,
                                      const std::vector<int64_t> &shape,const std::vector<int64_t> &vals,
                                      std::vector<std::vector<int32_t>> &i32_scratch) {
        if(dt==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return Ort::Value::CreateTensor<int64_t>(mem,const_cast<int64_t *>(vals.data()),vals.size(),shape.data(),shape.size());
        }
        require(dt==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,"TTS_UNEXPECTED_INT_DTYPE");
        i32_scratch.emplace_back(vals.size());
        auto &buf=i32_scratch.back();
        for(size_t i=0;i<vals.size();++i) buf[i]=static_cast<int32_t>(vals[i]);
        return Ort::Value::CreateTensor<int32_t>(mem,buf.data(),buf.size(),shape.data(),shape.size());
    }

    static Ort::Value make_float_tensor(const OrtMemoryInfo *mem,ONNXTensorElementDataType dt,
                                        const std::vector<int64_t> &shape,const std::vector<float> &vals,
                                        std::vector<std::vector<Ort::Float16_t>> &f16_scratch) {
        if(dt==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return Ort::Value::CreateTensor<float>(mem,const_cast<float *>(vals.data()),vals.size(),shape.data(),shape.size());
        }
        require(dt==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,"TTS_UNEXPECTED_FLOAT_DTYPE");
        f16_scratch.emplace_back(vals.size());
        auto &buf=f16_scratch.back();
        for(size_t i=0;i<vals.size();++i) buf[i]=Ort::Float16_t(vals[i]);
        return Ort::Value::CreateTensor<Ort::Float16_t>(mem,buf.data(),buf.size(),shape.data(),shape.size());
    }

    static std::vector<float> read_floats(const Ort::Value &val,std::vector<int64_t> *out_shape=nullptr) {
        auto info=val.GetTensorTypeAndShapeInfo();
        auto shape=info.GetShape();
        if(out_shape) *out_shape=shape;
        size_t n=info.GetElementCount();
        std::vector<float> out(n);
        if(n==0) return out;
        auto dt=info.GetElementType();
        if(dt==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float *src=val.GetTensorData<float>();
            std::memcpy(out.data(),src,n*sizeof(float));
        } else {
            require(dt==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,"TTS_UNEXPECTED_OUT_DTYPE");
            const Ort::Float16_t *src=val.GetTensorData<Ort::Float16_t>();
            for(size_t i=0;i<n;++i) out[i]=static_cast<float>(src[i]);
        }
        return out;
    }

    std::vector<Ort::Value> make_input_slots() const {
        std::vector<Ort::Value> slots;
        slots.reserve(in_names.size());
        for(size_t i=0;i<in_names.size();++i) slots.emplace_back(nullptr);
        return slots;
    }

    std::vector<Ort::Value> run_all(const std::vector<Ort::Value> &inputs) {
        return session.Run(Ort::RunOptions{nullptr},in_cstrs.data(),inputs.data(),inputs.size(),
                           out_cstrs.data(),out_cstrs.size());
    }
};

class MeloTTSOnnxModel final : public Model {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"deskpet_melo_tts"};
    OrtModule melo;
    OrtModule bert;
    bool has_bert=false;
    json bert_vocab;
    std::unordered_map<std::string,int64_t> sym_to_id;
    int64_t pad_id=0;
    int64_t kr_lang_id=4;
    int64_t kr_tone_start=11;
    int sample_rate=44100;

    static void append_utf8(uint32_t cp,std::string &out) {
        if(cp<0x80) out.push_back(static_cast<char>(cp));
        else if(cp<0x800) {
            out.push_back(static_cast<char>(0xC0|(cp>>6)));
            out.push_back(static_cast<char>(0x80|(cp&0x3F)));
        } else if(cp<0x10000) {
            out.push_back(static_cast<char>(0xE0|(cp>>12)));
            out.push_back(static_cast<char>(0x80|((cp>>6)&0x3F)));
            out.push_back(static_cast<char>(0x80|(cp&0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0|(cp>>18)));
            out.push_back(static_cast<char>(0x80|((cp>>12)&0x3F)));
            out.push_back(static_cast<char>(0x80|((cp>>6)&0x3F)));
            out.push_back(static_cast<char>(0x80|(cp&0x3F)));
        }
    }

    static std::vector<uint32_t> utf8_to_cps(const std::string &s) {
        std::vector<uint32_t> cps;
        for(size_t i=0;i<s.size();) {
            unsigned char c=static_cast<unsigned char>(s[i]);
            if((c&0x80)==0) { cps.push_back(c); i+=1; }
            else if((c&0xE0)==0xC0&&i+1<s.size()) {
                cps.push_back(((c&0x1F)<<6)|(static_cast<unsigned char>(s[i+1])&0x3F)); i+=2;
            } else if((c&0xF0)==0xE0&&i+2<s.size()) {
                cps.push_back(((c&0x0F)<<12)|((static_cast<unsigned char>(s[i+1])&0x3F)<<6)|(static_cast<unsigned char>(s[i+2])&0x3F)); i+=3;
            } else if((c&0xF8)==0xF0&&i+3<s.size()) {
                cps.push_back(((c&0x07)<<18)|((static_cast<unsigned char>(s[i+1])&0x3F)<<12)|((static_cast<unsigned char>(s[i+2])&0x3F)<<6)|(static_cast<unsigned char>(s[i+3])&0x3F)); i+=4;
            } else { i+=1; }
        }
        return cps;
    }

    static std::string int_to_sino_korean(uint64_t n) {
        if(n==0) return "영";
        static const char *digits[]={"","일","이","삼","사","오","육","칠","팔","구"};
        static const char *small_u[]={"","십","백","천"};
        static const char *big_u[]={"","만","억","조"};
        std::string res;
        int group=0;
        while(n>0) {
            uint64_t chunk=n%10000;
            n/=10000;
            if(chunk>0) {
                std::string part;
                for(int i=3;i>=0;--i) {
                    int div=1; for(int k=0;k<i;++k) div*=10;
                    int d=static_cast<int>((chunk/div)%10);
                    if(d==0) continue;
                    if(!(d==1&&i>0)) part+=digits[d];
                    part+=small_u[i];
                }
                if(chunk==1&&group==1&&n==0) part="";
                res=part+big_u[group]+res;
            }
            ++group;
        }
        return res;
    }

    static std::string normalize_korean_text(const std::string &input) {
        static const char *alpha_ko[]={
            "에이","비","씨","디","이","에프","지","에이치","아이","제이","케이","엘","엠",
            "엔","오","피","큐","알","에스","티","유","브이","더블유","엑스","와이","지"
        };
        std::string out;
        for(size_t i=0;i<input.size();) {
            unsigned char c=static_cast<unsigned char>(input[i]);
            if(std::isdigit(c)) {
                size_t j=i;
                while(j<input.size()&&std::isdigit(static_cast<unsigned char>(input[j]))) ++j;
                std::string num=input.substr(i,j-i);
                if(num.size()<=12&&!(num.size()>1&&num[0]=='0')) {
                    out+=int_to_sino_korean(std::stoull(num));
                } else {
                    static const char *dko[]={"영","일","이","삼","사","오","육","칠","팔","구"};
                    for(char ch:num) out+=dko[ch-'0'];
                }
                i=j;
            } else if(std::isalpha(c)) {
                char up=static_cast<char>(std::toupper(c));
                if(up>='A'&&up<='Z') out+=alpha_ko[up-'A'];
                ++i;
            } else {
                out.push_back(input[i]);
                ++i;
            }
        }
        return out;
    }

    struct Syllable { int L, V, T; };

    static uint32_t neutralize_coda_cp(int t) {
        // Map Hangul Jongseong (1..27) to 7 representative coda Jamo in symbols.json.
        switch(t) {
            case 1: case 2: case 3: case 24: return 0x11A8; // ᆨ
            case 4: case 5: case 6: return 0x11AB;          // ᆫ
            case 7: case 19: case 20: case 22: case 23: case 25: case 27: return 0x11AE; // ᆮ
            case 8: case 9: case 12: case 13: case 14: case 15: return 0x11AF;          // ᆯ
            case 10: case 16: return 0x11B7;                // ᆷ
            case 11: case 17: case 18: case 26: return 0x11B8; // ᆸ
            case 21: return 0x11BC;                         // ᆼ
            default: return 0;
        }
    }

    static int coda_to_choseong(int t) {
        // Returns Choseong index (0..18) when single/second coda liaises to ㅇ, else -1.
        switch(t) {
            case 1: return 0; case 2: return 1; case 4: return 2; case 7: return 3;
            case 8: return 5; case 16: return 6; case 17: return 7; case 19: return 9;
            case 20: return 10; case 22: return 12; case 23: return 14; case 24: return 15;
            case 25: return 16; case 26: return 17; default: return -1;
        }
    }

    static void apply_korean_phonology(std::vector<Syllable> &s) {
        for(size_t i=0;i+1<s.size();++i) {
            int &t1=s[i].T;
            int &l2=s[i+1].L;
            int v2=s[i+1].V;
            if(t1==0) continue;

            // 1) Palatalization (ㄷ/ㅌ + 이 -> ㅈ/ㅊ)
            if((t1==7||t1==25)&&l2==11&&(v2==20||v2==2||v2==6||v2==12||v2==17)) {
                l2=(t1==7)?12:14;
                t1=0;
                continue;
            }
            // 2) Aspiration with ㅎ
            if((t1==27||t1==6||t1==15)&&(l2==0||l2==3||l2==12)) {
                l2=(l2==0)?15:(l2==3)?16:14;
                t1=(t1==6)?4:(t1==15)?8:0;
                continue;
            }
            if(l2==18&&(t1==1||t1==24||t1==7||t1==17||t1==22||t1==23)) {
                l2=(t1==1||t1==24)?15:(t1==7)?16:(t1==17)?17:14;
                t1=0;
                continue;
            }
            // 3) Liaison before ㅇ (L2 == 11)
            if(l2==11) {
                if(t1==27) { t1=0; continue; } // ㅎ-deletion
                if(t1==3) { t1=1; l2=10; continue; }      // ㄳ -> ㄱ+ㅆ
                if(t1==5) { t1=4; l2=12; continue; }      // ㄵ -> ㄴ+ㅈ
                if(t1==6) { t1=4; continue; }             // ㄶ -> ㄴ
                if(t1==9) { t1=8; l2=0; continue; }       // ㄺ -> ㄹ+ㄱ
                if(t1==10) { t1=8; l2=6; continue; }      // ㄻ -> ㄹ+ㅁ
                if(t1==11) { t1=8; l2=7; continue; }      // ㄼ -> ㄹ+ㅂ
                if(t1==12) { t1=8; l2=10; continue; }     // ㄽ -> ㄹ+ㅆ
                if(t1==13) { t1=8; l2=16; continue; }     // ㄾ -> ㄹ+ㅌ
                if(t1==14) { t1=8; l2=17; continue; }     // ㄿ -> ㄹ+ㅍ
                if(t1==15) { t1=8; continue; }            // ㅀ -> ㄹ
                if(t1==18) { t1=17; l2=10; continue; }    // ㅄ -> ㅂ+ㅆ
                int moved=coda_to_choseong(t1);
                if(moved>=0) { l2=moved; t1=0; continue; }
            }
            // 4) Liquidization (ㄴ+ㄹ / ㄹ+ㄴ -> ㄹ+ㄹ)
            if(t1==4&&l2==5) { t1=8; continue; }
            if((t1==8||t1==15)&&l2==2) { l2=5; if(t1==15) t1=8; continue; }
            // 5) Nasalization
            bool stop_k=(t1==1||t1==2||t1==3||t1==9||t1==24);
            bool stop_t=(t1==7||t1==19||t1==20||t1==22||t1==23||t1==25||t1==27);
            bool stop_p=(t1==11||t1==14||t1==17||t1==18||t1==26);
            if(l2==2||l2==6) {
                if(stop_k) { t1=21; continue; }
                if(stop_t) { t1=4; continue; }
                if(stop_p) { t1=16; continue; }
            }
            if(l2==5) {
                if(t1==16||t1==21) { l2=2; continue; }
                if(stop_k) { t1=21; l2=2; continue; }
                if(stop_p) { t1=16; l2=2; continue; }
            }
            // 6) Tensification after obstruent codas
            if(stop_k||stop_t||stop_p) {
                if(l2==0) l2=1;
                else if(l2==3) l2=4;
                else if(l2==7) l2=8;
                else if(l2==9) l2=10;
                else if(l2==12) l2=13;
            }
        }
    }

    std::vector<int64_t> korean_text_to_phone_ids(const std::string &raw_text,
                                                  std::vector<std::pair<std::string,int>> *word_phones=nullptr) const {
        std::string text=normalize_korean_text(raw_text);
        auto cps=utf8_to_cps(text);
        std::vector<int64_t> phones;

        auto push_sym=[&](const std::string &sym) -> bool {
            auto it=sym_to_id.find(sym);
            if(it==sym_to_id.end()) return false;
            phones.push_back(it->second);
            return true;
        };

        for(size_t i=0;i<cps.size();) {
            uint32_t cp=cps[i];
            if(cp>=0xAC00&&cp<=0xD7A3) {
                std::vector<Syllable> run;
                std::string run_utf8;
                while(i<cps.size()&&cps[i]>=0xAC00&&cps[i]<=0xD7A3) {
                    uint32_t idx=cps[i]-0xAC00;
                    int l=static_cast<int>(idx/(21*28));
                    int v=static_cast<int>((idx%(21*28))/28);
                    int t=static_cast<int>(idx%28);
                    run.push_back({l,v,t});
                    append_utf8(cps[i],run_utf8);
                    ++i;
                }
                apply_korean_phonology(run);
                size_t before=phones.size();
                for(const auto &sy:run) {
                    std::string sj;
                    append_utf8(0x1100+static_cast<uint32_t>(sy.L),sj); push_sym(sj);
                    sj.clear(); append_utf8(0x1161+static_cast<uint32_t>(sy.V),sj); push_sym(sj);
                    if(sy.T>0) {
                        uint32_t ccp=neutralize_coda_cp(sy.T);
                        if(ccp) { sj.clear(); append_utf8(ccp,sj); push_sym(sj); }
                    }
                }
                if(word_phones) word_phones->emplace_back(run_utf8,static_cast<int>(phones.size()-before));
            } else {
                std::string ch;
                append_utf8(cp,ch);
                if(cp==' '||cp=='\t'||cp=='\n'||cp=='\r') {
                    // Preserve word spacing pause when consecutive.
                    if(!phones.empty()&&sym_to_id.count("SP")&&phones.back()!=sym_to_id.at("SP")) {
                        // MeloTTS uses SP or inter-word punctuation; keep minimal boundary.
                    }
                } else if(ch=="."||ch==","||ch=="!"||ch=="?"||ch=="-"||ch=="…") {
                    size_t before=phones.size();
                    if(push_sym(ch)&&word_phones) word_phones->emplace_back(ch,static_cast<int>(phones.size()-before));
                } else if(cp==':'||cp==';') {
                    size_t before=phones.size();
                    if(push_sym(",")&&word_phones) word_phones->emplace_back(",",static_cast<int>(phones.size()-before));
                }
                ++i;
            }
        }
        return phones;
    }

    std::vector<float> compute_korean_bert(const std::string &text,int64_t total_phones) {
        std::vector<float> out(static_cast<size_t>(768*total_phones),0.0f);
        if(!has_bert||total_phones<=0) return out;
        // Run Korean BERT WordPiece + distribute across phones when optional BERT ONNX is supplied.
        std::vector<std::pair<std::string,int>> wp;
        auto raw_phones=korean_text_to_phone_ids(text,&wp);
        if(wp.empty()||raw_phones.empty()) return out;

        std::vector<int64_t> input_ids;
        std::vector<int> token_phone_counts;
        int64_t cls_id=bert_vocab.value("[CLS]",int64_t(2));
        int64_t sep_id=bert_vocab.value("[SEP]",int64_t(3));
        int64_t unk_id=bert_vocab.value("[UNK]",int64_t(1));
        input_ids.push_back(cls_id);
        token_phone_counts.push_back(1); // initial blank

        for(const auto &item:wp) {
            auto cps=utf8_to_cps(item.first);
            std::vector<std::string> parts;
            for(uint32_t c:cps) { std::string s; append_utf8(c,s); parts.push_back(s); }
            std::vector<int64_t> sub_ids;
            size_t pos=0;
            while(pos<parts.size()) {
                bool found=false;
                for(size_t end=parts.size();end>pos;--end) {
                    std::string cand=pos?"##":"";
                    for(size_t j=pos;j<end;++j) cand+=parts[j];
                    if(bert_vocab.contains(cand)) { sub_ids.push_back(bert_vocab.at(cand).get<int64_t>()); pos=end; found=true; break; }
                }
                if(!found) { sub_ids={unk_id}; break; }
            }
            int k=static_cast<int>(sub_ids.size());
            int total_ph=item.second*2; // interspersed blanks
            int q=total_ph/std::max(1,k), rem=total_ph%std::max(1,k);
            for(int t=0;t<k;++t) {
                if(input_ids.size()>=510) break;
                input_ids.push_back(sub_ids[static_cast<size_t>(t)]);
                token_phone_counts.push_back(q+(t<rem?1:0));
            }
        }
        input_ids.push_back(sep_id);
        token_phone_counts.push_back(0);

        int64_t seq=static_cast<int64_t>(input_ids.size());
        std::vector<int64_t> mask(static_cast<size_t>(seq),1), types(static_cast<size_t>(seq),0);
        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto ins=bert.make_input_slots();
        std::vector<std::vector<int32_t>> i32;
        for(size_t i=0;i<bert.in_names.size();++i) {
            const auto &n=bert.in_names[i];
            const auto &vec=(n=="input_ids")?input_ids:(n=="attention_mask")?mask:types;
            ins[i]=OrtModule::make_int_tensor(mem,bert.in_dtypes[i],{1,seq},vec,i32);
        }
        auto outs=bert.run_all(ins);
        std::vector<int64_t> hshape;
        auto h=OrtModule::read_floats(outs[0],&hshape);
        if(hshape.size()<3||hshape[2]!=768) return out;

        int64_t col=0;
        for(size_t tok=0;tok<token_phone_counts.size()&&tok<static_cast<size_t>(seq);++tok) {
            int rep=token_phone_counts[tok];
            for(int r=0;r<rep&&col<total_phones;++r,++col) {
                for(int d=0;d<768;++d) {
                    out[static_cast<size_t>(d*total_phones+col)]=h[tok*768+static_cast<size_t>(d)];
                }
            }
        }
        return out;
    }

public:
    explicit MeloTTSOnnxModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","TTS_DEVICE_UNSUPPORTED");
        int threads=c.value("threads",2);
        melo.load(env,path("model"),dev,threads);

        auto sj=read_json(path("symbols"));
        const auto &smap=sj.at("symbol_to_id");
        for(auto it=smap.begin();it!=smap.end();++it) sym_to_id[it.key()]=it.value().get<int64_t>();
        std::string pad_sym=sj.value("pad",std::string("_"));
        pad_id=sym_to_id.count(pad_sym)?sym_to_id.at(pad_sym):0;
        if(sj.contains("language_id_map")&&sj["language_id_map"].contains("KR")) {
            kr_lang_id=sj["language_id_map"]["KR"].get<int64_t>();
        }
        if(sj.contains("language_tone_start_map")&&sj["language_tone_start_map"].contains("KR")) {
            kr_tone_start=sj["language_tone_start_map"]["KR"].get<int64_t>();
        }
        sample_rate=c.value("sampleRate",44100);

        const auto &paths=c.at("paths");
        if(paths.contains("bert")&&paths.at("bert").is_string()&&
           paths.contains("bert_tokenizer")&&paths.at("bert_tokenizer").is_string()) {
            std::string bpath=path("bert");
            std::string tpath=path("bert_tokenizer");
            if(speech_file_exists(bpath)&&speech_file_exists(tpath)) {
                bert.load(env,bpath,dev,threads);
                auto tj=read_json(tpath);
                bert_vocab=tj.at("model").at("vocab");
                has_bert=true;
            }
        }
    }
    ~MeloTTSOnnxModel() override { stop(); }

    json infer(Job &job) override {
        auto text=job.input.at("text").get<std::string>();
        require(!text.empty()&&text.size()<=8192,"TTS_TEXT_LIMIT");
        require(!job.cancelled.load(),"CANCELLED");

        auto raw_phones=korean_text_to_phone_ids(text);
        require(!raw_phones.empty(),"TTS_EMPTY_TOKENS");

        // Intersperse pad_id (add_blank = true) matching MeloTTS frontend.
        int64_t T=static_cast<int64_t>(raw_phones.size()*2+1);
        std::vector<int64_t> x(static_cast<size_t>(T),pad_id);
        std::vector<int64_t> tone(static_cast<size_t>(T),0);
        std::vector<int64_t> lang(static_cast<size_t>(T),0);
        for(size_t i=0;i<raw_phones.size();++i) {
            x[2*i+1]=raw_phones[i];
            tone[2*i+1]=kr_tone_start;
            lang[2*i+1]=kr_lang_id;
        }

        std::vector<int64_t> x_len{T}, sid{0};
        std::vector<float> bert_zeros(static_cast<size_t>(1024*T),0.0f);
        std::vector<float> ja_bert=compute_korean_bert(text,T);
        std::vector<float> noise_scale{0.6f}, length_scale{1.0f}, noise_scale_w{0.8f}, sdp_ratio{0.2f};

        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto ins=melo.make_input_slots();
        std::vector<std::vector<int32_t>> i32;
        std::vector<std::vector<Ort::Float16_t>> f16;

        auto scalar_shape=[&](size_t idx) -> std::vector<int64_t> {
            return melo.in_shapes[idx].empty()?std::vector<int64_t>{}:std::vector<int64_t>{1};
        };

        for(size_t idx=0;idx<melo.in_names.size();++idx) {
            const auto &name=melo.in_names[idx];
            if(name=="x") ins[idx]=OrtModule::make_int_tensor(mem,melo.in_dtypes[idx],{1,T},x,i32);
            else if(name=="x_lengths") ins[idx]=OrtModule::make_int_tensor(mem,melo.in_dtypes[idx],{1},x_len,i32);
            else if(name=="sid") ins[idx]=OrtModule::make_int_tensor(mem,melo.in_dtypes[idx],{1},sid,i32);
            else if(name=="tone") ins[idx]=OrtModule::make_int_tensor(mem,melo.in_dtypes[idx],{1,T},tone,i32);
            else if(name=="language") ins[idx]=OrtModule::make_int_tensor(mem,melo.in_dtypes[idx],{1,T},lang,i32);
            else if(name=="bert") ins[idx]=OrtModule::make_float_tensor(mem,melo.in_dtypes[idx],{1,1024,T},bert_zeros,f16);
            else if(name=="ja_bert") ins[idx]=OrtModule::make_float_tensor(mem,melo.in_dtypes[idx],{1,768,T},ja_bert,f16);
            else if(name=="noise_scale") ins[idx]=OrtModule::make_float_tensor(mem,melo.in_dtypes[idx],scalar_shape(idx),noise_scale,f16);
            else if(name=="length_scale") ins[idx]=OrtModule::make_float_tensor(mem,melo.in_dtypes[idx],scalar_shape(idx),length_scale,f16);
            else if(name=="noise_scale_w") ins[idx]=OrtModule::make_float_tensor(mem,melo.in_dtypes[idx],scalar_shape(idx),noise_scale_w,f16);
            else if(name=="sdp_ratio") ins[idx]=OrtModule::make_float_tensor(mem,melo.in_dtypes[idx],scalar_shape(idx),sdp_ratio,f16);
            else require(false,"TTS_MELO_UNSUPPORTED_INPUT: "+name);
        }

        require(!job.cancelled.load(),"CANCELLED");
        auto outs=melo.run_all(ins);
        require(!outs.empty(),"TTS_INFERENCE_FAILED");
        size_t o_idx=melo.out_idx.count("audio")?melo.out_idx.at("audio"):0;
        auto wav=OrtModule::read_floats(outs[o_idx]);
        require(!wav.empty(),"TTS_EMPTY_WAVEFORM");
        return {{"pcm",encode_pcm(wav.data(),wav.size())},{"sampleRate",sample_rate}};
    }
};

#if !defined(DESKPET_HAS_TOKENIZERS_CPP)
struct SimpleByteLevelBPE {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::string, int> ranks;
    std::string byte_utf8[256];
    int32_t unk_id = 0;

    static void append_utf8(uint32_t cp, std::string &out) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    void init_byte_table() {
        bool keep[256]{};
        for (int b = 33; b <= 126; ++b) keep[b] = true;
        for (int b = 161; b <= 172; ++b) keep[b] = true;
        for (int b = 174; b <= 255; ++b) keep[b] = true;
        uint32_t next_cp = 256;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp = keep[b] ? static_cast<uint32_t>(b) : next_cp++;
            byte_utf8[b].clear();
            append_utf8(cp, byte_utf8[b]);
        }
    }

    void add_merge_line(const std::string &line, int rank) {
        if (line.empty() || line[0] == '#') return;
        size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size()) return;
        std::string key = line.substr(0, sp);
        key.push_back('\n');
        key += line.substr(sp + 1);
        ranks.emplace(std::move(key), rank);
    }

    bool load_json_str(const std::string &blob) {
        init_byte_table();
        auto j = json::parse(blob);
        const auto &model = j.at("model");
        for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
            vocab.emplace(it.key(), it.value().get<int32_t>());
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (const auto &at : j["added_tokens"]) {
                if (at.contains("content") && at.contains("id")) {
                    vocab.emplace(at["content"].get<std::string>(), at["id"].get<int32_t>());
                }
            }
        }
        const auto &merges = model.at("merges");
        int rank = 0;
        for (const auto &m : merges) {
            if (m.is_string()) {
                add_merge_line(m.get<std::string>(), rank++);
            } else if (m.is_array() && m.size() == 2) {
                std::string key = m[0].get<std::string>();
                key.push_back('\n');
                key += m[1].get<std::string>();
                ranks.emplace(std::move(key), rank++);
            }
        }
        if (vocab.count("<|endoftext|>")) unk_id = vocab.at("<|endoftext|>");
        return !vocab.empty();
    }

    bool load_bpe_files(const std::string &vocab_blob, const std::string &merges_blob) {
        init_byte_table();
        auto vj = json::parse(vocab_blob);
        for (auto it = vj.begin(); it != vj.end(); ++it) {
            vocab.emplace(it.key(), it.value().get<int32_t>());
        }
        int rank = 0;
        size_t pos = 0;
        while (pos < merges_blob.size()) {
            size_t nl = merges_blob.find('\n', pos);
            if (nl == std::string::npos) nl = merges_blob.size();
            std::string line = merges_blob.substr(pos, nl - pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            add_merge_line(line, rank++);
            pos = nl + 1;
        }
        if (vocab.count("<|endoftext|>")) unk_id = vocab.at("<|endoftext|>");
        return !vocab.empty();
    }

    static bool is_ascii_letter(unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    static bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }

    static std::vector<std::string> pretokenize(const std::string &text) {
        std::vector<std::string> chunks;
        size_t i = 0;
        const size_t n = text.size();
        auto is_letter_at = [&](size_t idx) -> bool {
            unsigned char c = static_cast<unsigned char>(text[idx]);
            return is_ascii_letter(c) || c >= 0x80;
        };
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == '\r' || c == '\n') {
                size_t j = i + 1;
                while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
                chunks.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_ascii_digit(c)) {
                chunks.push_back(text.substr(i, 1));
                i += 1;
            } else if (is_letter_at(i)) {
                size_t j = i;
                while (j < n && is_letter_at(j)) ++j;
                chunks.push_back(text.substr(i, j - i));
                i = j;
            } else if (c == ' ' && i + 1 < n) {
                unsigned char next = static_cast<unsigned char>(text[i + 1]);
                if (is_letter_at(i + 1)) {
                    size_t j = i + 1;
                    while (j < n && is_letter_at(j)) ++j;
                    chunks.push_back(text.substr(i, j - i));
                    i = j;
                } else if (next != ' ' && next != '\t' && next != '\r' && next != '\n' && !is_ascii_digit(next)) {
                    size_t j = i + 1;
                    while (j < n) {
                        unsigned char nj = static_cast<unsigned char>(text[j]);
                        if (nj >= 0x80 || nj == ' ' || nj == '\t' || nj == '\r' || nj == '\n' || is_ascii_letter(nj) || is_ascii_digit(nj)) break;
                        ++j;
                    }
                    chunks.push_back(text.substr(i, j - i));
                    i = j;
                } else {
                    size_t j = i + 1;
                    while (j < n && (text[j] == ' ' || text[j] == '\t') &&
                           !(j + 1 < n && text[j] == ' ' && !std::isspace(static_cast<unsigned char>(text[j + 1])) && !is_ascii_digit(static_cast<unsigned char>(text[j + 1])))) {
                        ++j;
                    }
                    chunks.push_back(text.substr(i, j - i));
                    i = j;
                }
            } else if (c == ' ' || c == '\t') {
                size_t j = i + 1;
                while (j < n && (text[j] == ' ' || text[j] == '\t')) ++j;
                chunks.push_back(text.substr(i, j - i));
                i = j;
            } else {
                size_t j = i + 1;
                while (j < n) {
                    unsigned char nj = static_cast<unsigned char>(text[j]);
                    if (nj >= 0x80 || nj == ' ' || nj == '\t' || nj == '\r' || nj == '\n' || is_ascii_letter(nj) || is_ascii_digit(nj)) break;
                    ++j;
                }
                chunks.push_back(text.substr(i, j - i));
                i = j;
            }
        }
        return chunks;
    }

    std::vector<int32_t> Encode(const std::string &text) const {
        std::vector<int32_t> ids;
        for (const auto &chunk : pretokenize(text)) {
            std::vector<std::string> word;
            word.reserve(chunk.size());
            for (unsigned char b : chunk) word.push_back(byte_utf8[b]);
            while (word.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                std::string best_a, best_b;
                for (size_t i = 0; i + 1 < word.size(); ++i) {
                    std::string key = word[i];
                    key.push_back('\n');
                    key += word[i + 1];
                    auto it = ranks.find(key);
                    if (it != ranks.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_a = word[i];
                        best_b = word[i + 1];
                    }
                }
                if (best_rank == std::numeric_limits<int>::max()) break;
                std::vector<std::string> next_word;
                next_word.reserve(word.size());
                for (size_t i = 0; i < word.size();) {
                    if (i + 1 < word.size() && word[i] == best_a && word[i + 1] == best_b) {
                        next_word.push_back(best_a + best_b);
                        i += 2;
                    } else {
                        next_word.push_back(std::move(word[i]));
                        ++i;
                    }
                }
                word = std::move(next_word);
            }
            for (const auto &tok : word) {
                auto it = vocab.find(tok);
                ids.push_back(it != vocab.end() ? it->second : unk_id);
            }
        }
        return ids;
    }
};
#endif

class Qwen3TTSOnnxModel final : public Model {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"deskpet_tts"};
    OrtModule text_embed, codec_embed, residual_embed, talker, code_predictor, tok_decoder;
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
#else
    std::unique_ptr<SimpleByteLevelBPE> tokenizer;
#endif
    json cfg;
    int64_t hidden_size=1024, vocab_size=3072, n_groups=16;
    int64_t n_kv=8, head_dim=128;
    int64_t im_start_id=151644, assistant_id=77091, newline_id=198;
    int64_t tts_bos_id=151672, tts_eos_id=151673, tts_pad_id=151671;
    int64_t codec_bos_id=2149, codec_eos_id=2150, codec_pad_id=2148;
    int64_t codec_think_id=2154, codec_nothink_id=2155, codec_think_bos_id=2156, codec_think_eos_id=2157;
    int64_t language_id=-1, speaker_id=-1;
    int max_new_tokens=2048;

    bool has_kv_cache=false;
    bool pad_initial_kv=false;
    std::vector<std::string> past_names, present_names;

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

    std::vector<float> embed_text(const OrtMemoryInfo *mem,const std::vector<int64_t> &ids) {
        auto ins=text_embed.make_input_slots();
        std::vector<std::vector<int32_t>> i32;
        size_t idx=text_embed.require_input("text_ids");
        ins[idx]=OrtModule::make_int_tensor(mem,text_embed.in_dtypes[idx],{1,static_cast<int64_t>(ids.size())},ids,i32);
        auto outs=text_embed.run_all(ins);
        return OrtModule::read_floats(outs[text_embed.require_output("text_embeds")]);
    }
    std::vector<float> embed_codec(const OrtMemoryInfo *mem,const std::vector<int64_t> &ids) {
        auto ins=codec_embed.make_input_slots();
        std::vector<std::vector<int32_t>> i32;
        size_t idx=codec_embed.require_input("codec_ids");
        ins[idx]=OrtModule::make_int_tensor(mem,codec_embed.in_dtypes[idx],{1,static_cast<int64_t>(ids.size())},ids,i32);
        auto outs=codec_embed.run_all(ins);
        return OrtModule::read_floats(outs[codec_embed.require_output("codec_embeds")]);
    }
    std::vector<float> run_residual_embed(const OrtMemoryInfo *mem,const std::vector<int64_t> &codes16) {
        auto ins=residual_embed.make_input_slots();
        std::vector<std::vector<int32_t>> i32;
        size_t idx=residual_embed.require_input("codec_ids");
        ins[idx]=OrtModule::make_int_tensor(mem,residual_embed.in_dtypes[idx],{1,n_groups},codes16,i32);
        auto outs=residual_embed.run_all(ins);
        return OrtModule::read_floats(outs[residual_embed.require_output("step_embed")]);
    }
    std::vector<float> run_code_predictor(const OrtMemoryInfo *mem,const std::vector<float> &last_hidden,
                                          const std::vector<int64_t> &codes16,std::vector<int64_t> &out_dims) {
        auto ins=code_predictor.make_input_slots();
        std::vector<std::vector<int32_t>> i32;
        std::vector<std::vector<Ort::Float16_t>> f16;
        size_t ih=code_predictor.require_input("talker_hidden");
        size_t ic=code_predictor.require_input("codec_ids");
        ins[ih]=OrtModule::make_float_tensor(mem,code_predictor.in_dtypes[ih],{1,hidden_size},last_hidden,f16);
        ins[ic]=OrtModule::make_int_tensor(mem,code_predictor.in_dtypes[ic],{1,n_groups},codes16,i32);
        auto outs=code_predictor.run_all(ins);
        return OrtModule::read_floats(outs[code_predictor.require_output("group_logits")],&out_dims);
    }

public:
    explicit Qwen3TTSOnnxModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","TTS_DEVICE_UNSUPPORTED");

        cfg=read_json(path("config"));
        const auto &tc=cfg.at("talker_config");
        hidden_size=tc.at("hidden_size").get<int64_t>();
        vocab_size=tc.at("vocab_size").get<int64_t>();
        n_groups=tc.at("num_code_groups").get<int64_t>();
        n_kv=tc.value("num_key_value_heads",8);
        head_dim=tc.value("head_dim",128);
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
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
            tokenizer=tokenizers::Tokenizer::FromBlobJSON(read_text_file(tok_path));
#else
            auto bpe=std::make_unique<SimpleByteLevelBPE>();
            if(bpe->load_json_str(read_text_file(tok_path))) tokenizer=std::move(bpe);
#endif
        } else {
            std::string dir=parent_dir(path("config"));
            std::string vocab_path=paths.contains("vocab")&&paths.at("vocab").is_string()?paths.at("vocab").get<std::string>():(dir+"/vocab.json");
            std::string merges_path=paths.contains("merges")&&paths.at("merges").is_string()?paths.at("merges").get<std::string>():(dir+"/merges.txt");
#if defined(DESKPET_HAS_TOKENIZERS_CPP)
            tokenizer=tokenizers::Tokenizer::FromBlobByteLevelBPE(read_text_file(vocab_path),read_text_file(merges_path));
#else
            auto bpe=std::make_unique<SimpleByteLevelBPE>();
            if(bpe->load_bpe_files(read_text_file(vocab_path),read_text_file(merges_path))) tokenizer=std::move(bpe);
#endif
        }
        require(bool(tokenizer),"TTS_TOKENIZER_LOAD_FAILED");
        auto nl=tokenizer->Encode("\n");
        require(!nl.empty(),"TTS_NEWLINE_TOKEN_FAILED");
        newline_id=nl[0];

        int threads=c.value("threads",2);
        text_embed.load(env,path("text_embed"),dev,threads);
        codec_embed.load(env,path("codec_embed"),dev,threads);
        residual_embed.load(env,path("residual_embed"),dev,threads);
        talker.load(env,path("talker"),dev,threads);
        code_predictor.load(env,path("code_predictor"),dev,threads);
        tok_decoder.load(env,path("tok_decoder"),dev,threads);

        for(const auto &name:talker.in_names) {
            if(name!="inputs_embeds"&&name!="position_ids"&&name!="attention_mask") past_names.push_back(name);
        }
        for(const auto &name:talker.out_names) {
            if(name!="logits"&&name!="hidden_states") present_names.push_back(name);
        }
        has_kv_cache=!past_names.empty();
        pad_initial_kv=has_kv_cache&&(dev=="metal"||dev=="coreml");
        if(has_kv_cache) {
            require(past_names.size()==present_names.size(),"TTS_KV_TENSOR_MISMATCH");
            std::sort(past_names.begin(),past_names.end(),natural_name_less);
            std::sort(present_names.begin(),present_names.end(),natural_name_less);
        }
    }
    ~Qwen3TTSOnnxModel() override { stop(); }

    json infer(Job &job) override {
        auto text=job.input.at("text").get<std::string>();
        require(!text.empty()&&text.size()<=8192,"TTS_TEXT_LIMIT");
        require(!job.cancelled.load(),"CANCELLED");

        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);

        // 0) Tokenize text and fetch fixed special token embeddings.
        auto body_raw=tokenizer->Encode(text);
        require(!body_raw.empty(),"TTS_EMPTY_TOKENS");
        std::vector<int64_t> body_ids(body_raw.begin(),body_raw.end());

        auto spec=embed_text(mem,{tts_bos_id,tts_eos_id,tts_pad_id});
        std::vector<float> bos_e(spec.begin(),spec.begin()+hidden_size);
        std::vector<float> eos_e(spec.begin()+hidden_size,spec.begin()+2*hidden_size);
        std::vector<float> pad_e(spec.begin()+2*hidden_size,spec.begin()+3*hidden_size);
        auto role=embed_text(mem,{im_start_id,assistant_id,newline_id});

        // 1) Language + speaker + codec prefix embeddings.
        std::vector<int64_t> codec_prefill;
        if(language_id<0) codec_prefill={codec_nothink_id,codec_think_bos_id,codec_think_eos_id};
        else codec_prefill={codec_think_id,codec_think_bos_id,language_id,codec_think_eos_id};
        auto codec0=embed_codec(mem,codec_prefill);
        auto codec1=embed_codec(mem,{codec_pad_id,codec_bos_id});
        std::vector<float> codec_input=std::move(codec0);
        if(speaker_id>=0) {
            auto spk_e=embed_codec(mem,{speaker_id});
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
        auto text_body=embed_text(mem,body_ids);
        auto codec_pads=embed_codec(mem,pad_ids);
        for(size_t i=0;i<body_ids.size();++i) {
            for(int64_t h=0;h<hidden_size;++h) {
                talker_in.push_back(text_body[i*hidden_size+h]+codec_pads[i*hidden_size+h]);
            }
        }
        for(int64_t h=0;h<hidden_size;++h) {
            talker_in.push_back(eos_e[static_cast<size_t>(h)]+codec_pads[body_ids.size()*hidden_size+h]);
        }
        auto codec_bos_e=embed_codec(mem,{codec_bos_id});
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
        std::vector<Ort::Value> kv_past_vals;
        std::vector<uint8_t> empty_kv_dummy(static_cast<size_t>(n_kv*head_dim*4),0);

        size_t i_emb=talker.require_input("inputs_embeds");
        size_t i_pos=talker.require_input("position_ids");
        size_t i_msk=talker.require_input("attention_mask");
        size_t o_log=talker.require_output("logits");
        size_t o_hid=talker.require_output("hidden_states");

        for(int step=0;step<max_new_tokens;++step) {
            require(!job.cancelled.load(),"CANCELLED");
            std::vector<int64_t> pos(static_cast<size_t>(3*cur_seq));
            for(int row=0;row<3;++row) {
                for(int64_t c=0;c<cur_seq;++c) pos[static_cast<size_t>(row*cur_seq+c)]=total_pos+c;
            }
            int64_t kv_init_len=pad_initial_kv?1:0;
            int64_t kv_in_len=(step==0)?kv_init_len:kv_past_len;
            int64_t mask_len=kv_in_len+cur_seq;
            std::vector<int64_t> mask(static_cast<size_t>(mask_len),1);
            if(pad_initial_kv&&mask_len>0) mask[0]=0;

            auto talker_ins=talker.make_input_slots();
            std::vector<std::vector<int32_t>> i32;
            std::vector<std::vector<Ort::Float16_t>> f16;
            talker_ins[i_emb]=OrtModule::make_float_tensor(mem,talker.in_dtypes[i_emb],{1,cur_seq,hidden_size},cur_embeds,f16);
            talker_ins[i_pos]=OrtModule::make_int_tensor(mem,talker.in_dtypes[i_pos],{3,1,cur_seq},pos,i32);
            talker_ins[i_msk]=OrtModule::make_int_tensor(mem,talker.in_dtypes[i_msk],{1,mask_len},mask,i32);

            if(has_kv_cache) {
                if(step==0) {
                    kv_past_vals.clear();
                    kv_past_vals.reserve(past_names.size());
                    for(const auto &pname:past_names) {
                        size_t pk=talker.require_input(pname);
                        const auto &ps=talker.in_shapes[pk];
                        int64_t kv_h=(ps.size()>=2&&ps[1]>0)?ps[1]:n_kv;
                        int64_t kv_d=(ps.size()>=4&&ps[3]>0)?ps[3]:head_dim;
                        std::vector<int64_t> kv_shape{1,kv_h,kv_init_len,kv_d};
                        size_t elem_sz=(talker.in_dtypes[pk]==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)?sizeof(float):sizeof(uint16_t);
                        size_t kv_bytes=static_cast<size_t>(kv_init_len*kv_h*kv_d)*elem_sz;
                        if(empty_kv_dummy.size()<kv_bytes) empty_kv_dummy.assign(kv_bytes,0);
                        kv_past_vals.push_back(Ort::Value::CreateTensor(
                            mem,empty_kv_dummy.data(),kv_bytes,kv_shape.data(),kv_shape.size(),talker.in_dtypes[pk]));
                    }
                }
                for(size_t k=0;k<past_names.size();++k) {
                    size_t pk=talker.require_input(past_names[k]);
                    talker_ins[pk]=std::move(kv_past_vals[k]);
                }
            }

            auto outputs=talker.run_all(talker_ins);
            auto logits=OrtModule::read_floats(outputs[o_log]);
            auto hidden=OrtModule::read_floats(outputs[o_hid]);

            if(has_kv_cache) {
                for(size_t k=0;k<present_names.size();++k) {
                    size_t ok=talker.require_output(present_names[k]);
                    kv_past_vals[k]=std::move(outputs[ok]);
                }
                auto pres_shape=kv_past_vals[0].GetTensorTypeAndShapeInfo().GetShape();
                require(pres_shape.size()>=3,"TTS_KV_SHAPE_INVALID");
                kv_past_len=pres_shape[2];
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
                std::vector<int64_t> gdims;
                auto gl=run_code_predictor(mem,last_hidden,codes16,gdims);
                require(gdims.size()>=3,"TTS_CODE_PREDICTOR_SHAPE_INVALID");
                int64_t sub_vocab=gdims[2];
                const float *row_logits=gl.data()+static_cast<size_t>((j-1)*sub_vocab);
                codes16[static_cast<size_t>(j)]=sample_token(row_logits,sub_vocab,true,top_k,top_p,temperature,rng);
            }
            all_codes.insert(all_codes.end(),codes16.begin(),codes16.end());

            auto next_emb=run_residual_embed(mem,codes16);
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
        size_t i_dec=tok_decoder.require_input("audio_codes");
        size_t o_wav=tok_decoder.require_output("waveform");
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
            auto dec_ins=tok_decoder.make_input_slots();
            std::vector<std::vector<int32_t>> i32;
            dec_ins[i_dec]=OrtModule::make_int_tensor(mem,tok_decoder.in_dtypes[i_dec],{1,DEC_FRAMES,n_groups},chunk,i32);
            auto dec_outs=tok_decoder.run_all(dec_ins);
            auto wav=OrtModule::read_floats(dec_outs[o_wav]);
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
static std::vector<std::string> utf8parts(const std::string &s) {
    std::vector<std::string> out;for(size_t i=0;i<s.size();) {auto c=uint8_t(s[i]);size_t n=c<128?1:(c&0xe0)==0xc0?2:(c&0xf0)==0xe0?3:4;
        require(i+n<=s.size(),"INVALID_UTF8");out.push_back(s.substr(i,n));i+=n;}return out;
}
class EmotionModel final : public Model {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"deskpet"};Ort::Session session{nullptr};
    json vocab;std::vector<std::string> inputNames,outputNames;
public:
    explicit EmotionModel(const json &c):Model(c) {
        std::string dev=c.at("device").get<std::string>();
        require(dev=="cpu"||dev=="cuda"||dev=="metal"||dev=="coreml","EMOTION_DEVICE_UNSUPPORTED");
        int threads=c.value("threads",2);
        session=OrtModule::create_session(env,path("model"),dev,threads);
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
    require(device=="cpu"||device=="cuda"||device=="metal"||device=="coreml","SPEECH_DEVICE_UNSUPPORTED");
    if(role=="stt"&&backend=="whisper.cpp") return std::make_shared<WhisperModel>(c);
#ifdef DESKPET_SPEECH
    if(role=="stt"&&backend=="sherpa-onnx") {
        const auto &p=c.at("paths");
        if(p.contains("joiner")&&p.contains("tokens")) return std::make_shared<ZipformerModel>(c);
        return std::make_shared<Qwen3ASRModel>(c);
    }
    if(role=="vad"&&backend=="sherpa-onnx") return std::make_shared<VADModel>(c);
    if(role=="tts"&&backend=="sherpa-onnx") return std::make_shared<TTSModel>(c);
    if(role=="tts"&&backend=="onnxruntime") {
        const auto &p=c.at("paths");
        if(p.contains("symbols")) return std::make_shared<MeloTTSOnnxModel>(c);
        return std::make_shared<Qwen3TTSOnnxModel>(c);
    }
    if(role=="emotion"&&backend=="onnxruntime") return std::make_shared<EmotionModel>(c);
#endif
    if(role=="tts"&&backend=="onnxruntime") throw std::runtime_error("ONNXRUNTIME_TTS_NOT_AVAILABLE");
    require((role=="stt"&&(backend=="sherpa-onnx"||backend=="whisper.cpp")) ||
            ((role=="vad"||role=="tts")&&backend=="sherpa-onnx") ||
            (role=="tts"&&backend=="onnxruntime") ||
            (role=="emotion"&&backend=="onnxruntime"),"BACKEND_UNSUPPORTED");
    throw std::runtime_error("NATIVE_DEPENDENCY_MISSING: "+role);
}
