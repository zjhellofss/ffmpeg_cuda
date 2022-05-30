#pragma once

class GpuFrameAdapter {
 public:
  explicit GpuFrameAdapter() {

  }
  GpuFrameAdapter(const GpuFrameAdapter &) = delete;
  GpuFrameAdapter &operator=(const GpuFrameAdapter &adapter) = delete;

  std::optional<AVPixelFormat> GetPixelFormat(AVFrame *cu_frame);
  std::optional<cv::cuda::GpuMat> ConvertFrame(AVFrame *cu_frame);
  void SetStream(cudaStream_t stream);

 private:
  cudaStream_t stream_ = nullptr;
};

class GpuDecoder {
 public:
  explicit GpuDecoder() : type_(AV_HWDEVICE_TYPE_CUDA) {
  }
  GpuDecoder(const GpuDecoder &) = delete;
  GpuDecoder &operator=(const GpuDecoder &) = delete;
  GpuDecoder &operator=(const GpuDecoder &&) = delete;

  ~GpuDecoder() {
    if (this->decoder_ctx_ != nullptr) {
      avcodec_free_context(&this->decoder_ctx_);
      this->decoder_ctx_ = nullptr;
    }
    if (this->input_ctx_ != nullptr) {
      avformat_close_input(&this->input_ctx_);
      this->input_ctx_ = nullptr;
    }
    if (hw_device_ctx_ != nullptr) {
      av_buffer_unref(&hw_device_ctx_);
      hw_device_ctx_ = nullptr;
    }
    if (this->stream_) {
      cudaStreamDestroy(this->stream_);
      this->stream_ = nullptr;
    }
  }
  void Init(AVFormatContext *input_ctx);
  std::optional<cv::cuda::GpuMat> ConvertFrame(AVFrame *cu_frame);

  AVCodecContext *GetAVCodecContext() const {
    return this->decoder_ctx_;
  }

  int GetVideoStream() const {
    return this->video_stream_;
  }
  static AVPixelFormat hw_pix_fmt_;

 private:
  void FindVideoStream();
  void InitHWDecode();
  int HwDecoderInit(AVCodecContext *ctx, AVHWDeviceType type);

  static AVPixelFormat GetHwFormat(AVCodecContext *ctx,
                                   const enum AVPixelFormat *pix_fmts);
 private:
  cudaStream_t stream_ = nullptr;
  GpuFrameAdapter adapter_;
 private:
  int video_stream_ = -1;
  enum AVHWDeviceType type_;
  AVCodecContext *decoder_ctx_ = nullptr;
  const AVCodec *decoder_ = nullptr;
  AVBufferRef *hw_device_ctx_ = nullptr;
  AVFormatContext *input_ctx_ = nullptr;
};


