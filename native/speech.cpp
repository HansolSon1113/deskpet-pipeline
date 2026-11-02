#include "runtime.hpp"
#include "whisper.h"
#include <algorithm>
#include <cmath>
#ifdef DESKPET_SPEECH
#include "sherpa-onnx/c-api/c-api.h"
#include "onnxruntime_cxx_api.h"
#endif
namespace {
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
class VADModel final : public Model {
    const SherpaOnnxVoiceActivityDetector *vad=nullptr;
    bool speaking=false;
public:
    explicit VADModel(const json &c):Model(c) {
        auto file=path("model");SherpaOnnxVadModelConfig cfg{};cfg.silero_vad.model=file.c_str();
        cfg.silero_vad.threshold=0.5f;cfg.silero_vad.min_silence_duration=0.5f;cfg.silero_vad.min_speech_duration=0.1f;
        cfg.silero_vad.window_size=512;cfg.silero_vad.max_speech_duration=29;cfg.sample_rate=16000;
        cfg.num_threads=1;cfg.provider="cpu";vad=SherpaOnnxCreateVoiceActivityDetector(&cfg,30);
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
        auto p=config.at("paths");std::vector<std::string> files;
        for(auto key:{"duration_predictor","text_encoder","vector_estimator","vocoder","tts_json","unicode_indexer","voice_style"})files.push_back(p.at(key));
        SherpaOnnxOfflineTtsConfig cfg{};auto &s=cfg.model.supertonic;
        s.duration_predictor=files[0].c_str();s.text_encoder=files[1].c_str();s.vector_estimator=files[2].c_str();
        s.vocoder=files[3].c_str();s.tts_json=files[4].c_str();s.unicode_indexer=files[5].c_str();s.voice_style=files[6].c_str();
        cfg.model.num_threads=c.value("threads",2);cfg.model.provider="cpu";cfg.max_num_sentences=1;cfg.silence_scale=1;
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
static std::vector<std::string> utf8parts(const std::string &s) {
    std::vector<std::string> out;for(size_t i=0;i<s.size();) {auto c=uint8_t(s[i]);size_t n=c<128?1:(c&0xe0)==0xc0?2:(c&0xf0)==0xe0?3:4;
        require(i+n<=s.size(),"INVALID_UTF8");out.push_back(s.substr(i,n));i+=n;}return out;
}
class EmotionModel final : public Model {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR,"deskpet"};Ort::Session session{nullptr};
    json vocab;std::vector<std::string> inputNames,outputNames;
public:
    explicit EmotionModel(const json &c):Model(c) {
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
    require(c.at("device")=="cpu","CPU_MODEL_CONFIGURATION_REQUIRED");
    std::string role=c.at("role"),backend=c.at("backend");
    require((role=="stt"&&backend=="whisper.cpp") || ((role=="vad"||role=="tts")&&backend=="sherpa-onnx") ||
            (role=="emotion"&&backend=="onnxruntime"),"BACKEND_UNSUPPORTED");
    if(role=="stt")return std::make_shared<WhisperModel>(c);
#ifdef DESKPET_SPEECH
    if(role=="vad")return std::make_shared<VADModel>(c);
    if(role=="tts")return std::make_shared<TTSModel>(c);
    if(role=="emotion")return std::make_shared<EmotionModel>(c);
#endif
    throw std::runtime_error("NATIVE_DEPENDENCY_MISSING: "+role);
}
