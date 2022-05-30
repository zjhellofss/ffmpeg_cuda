#include <cstdio>
// opencv
#include "opencv2/opencv.hpp"
//cuda
#include "cuda_runtime_api.h"
#include "help_cuda.h"
//glog
#include "glog/logging.h"
//ffmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "hw_decode.h"

#define FF_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))

std::optional<cv::cuda::GpuMat> GpuFrameAdapter::ConvertFrame(AVFrame *cu_frame) {
  cv::cuda::GpuMat gpu_mat;
  if (cu_frame == nullptr) {
    LOG(ERROR) << "Frame is null";
    return std::nullopt;
  }
  AVPixelFormat hw_format = static_cast<AVPixelFormat>(cu_frame->format);
  if (hw_format != AVPixelFormat::AV_PIX_FMT_CUDA) {
    LOG(ERROR) << "Frame pixel format is not AV_PIX_FMT_CUDA";
    return std::nullopt;
  }
  auto pixel_format_opt = this->GetPixelFormat(cu_frame);
  if (pixel_format_opt) {
    auto pixel_format = *pixel_format_opt;
    if (pixel_format != AVPixelFormat::AV_PIX_FMT_NV12) {
      LOG(WARNING) << "Decode pixel format is not NV12";
    }
  }

  cv::cuda::createContinuous(cu_frame->height * 3 / 2, cu_frame->width, CV_8UC1, gpu_mat);
  if (gpu_mat.empty()) {
    return std::nullopt;
  }

  int copy_offset = 0;
  if (this->stream_ == nullptr) {
    LOG(FATAL) << "Cuda stream is empty";
    return std::nullopt;
  }

  for (int i = 0; i < FF_ARRAY_ELEMS(cu_frame->data) && cu_frame->data[i]; i++) {
    int src_pitch = cu_frame->linesize[i];
    int copy_width = cu_frame->width;
    int desc_pitch = copy_width;
    int copy_height = cu_frame->height >> ((i == 0 || i == 3) ? 0 : 1);
    cudaStream_t used_stream = this->stream_;
    checkCudaErrors(cudaMemcpy2DAsync(gpu_mat.data + copy_offset,
                                      desc_pitch,
                                      cu_frame->data[i],
                                      src_pitch,
                                      copy_width,
                                      copy_height,
                                      cudaMemcpyDeviceToDevice, used_stream));
    copy_offset += copy_width * copy_height;
  }
  return gpu_mat;
}

std::optional<AVPixelFormat> GpuFrameAdapter::GetPixelFormat(AVFrame *cu_frame) {
  AVPixelFormat *formats;
  int ret = av_hwframe_transfer_get_formats(cu_frame->hw_frames_ctx,
                                            AV_HWFRAME_TRANSFER_DIRECTION_FROM,
                                            &formats, 0);
  if (ret < 0) {
    if (formats != nullptr)
      av_freep(&formats);
    return std::nullopt;
  }
  AVPixelFormat format = formats[0];
  av_freep(&formats);
  return format;
}

void GpuFrameAdapter::SetStream(cudaStream_t stream) {
  if (stream == nullptr) {
    LOG(FATAL) << "Set stream is nullptr";
  }
  this->stream_ = stream;
}

void GpuDecoder::Init(AVFormatContext *input_ctx) {
  if (input_ctx == nullptr) {
    LOG(FATAL) << "Input format context is empty";
  }
  this->input_ctx_ = input_ctx;
  FindVideoStream();
  InitHWDecode();
  //create Cuda stream
  checkCudaErrors(cudaStreamCreate(&this->stream_));
  this->adapter_.SetStream(this->stream_);
}

int GpuDecoder::HwDecoderInit(AVCodecContext *ctx, AVHWDeviceType type) {
  int err;
  err = av_hwdevice_ctx_create(&hw_device_ctx_, type,
                               nullptr, nullptr, 0);
  if (err != 0) {
    return err;
  }
  ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  return err;
}

AVPixelFormat  GpuDecoder::hw_pix_fmt_ = AVPixelFormat::AV_PIX_FMT_NONE;

void GpuDecoder::FindVideoStream() {
  if (avformat_find_stream_info(input_ctx_, nullptr) < 0) {
    char error_buf[512] = {0};
    snprintf(error_buf, 512, "Cannot find input stream information");
    LOG(FATAL) << error_buf;
  }

  int ret = av_find_best_stream(input_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, &this->decoder_, 0);
  if (ret < 0) {
    char error_buf[512] = {0};
    snprintf(error_buf, 512, "Cannot find input stream information");
    LOG(FATAL) << error_buf;
  }
  this->video_stream_ = ret;
}

void GpuDecoder::InitHWDecode() {
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(this->decoder_, i);
    if (!config) {
      char error_buf[512] = {0};
      snprintf(error_buf, 512, "Decoder %s does not support device type %s",
               this->decoder_->name, av_hwdevice_get_type_name(this->type_));
      LOG(FATAL) << error_buf;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
        config->device_type == this->type_) {
      GpuDecoder::hw_pix_fmt_ = config->pix_fmt;
      break;
    }
  }

  if (this->decoder_ctx_ = avcodec_alloc_context3(decoder_); this->decoder_ctx_ == nullptr) {
    LOG(FATAL) << "Avcodec context init error";
  }
  AVStream *video = input_ctx_->streams[this->video_stream_];
  if (avcodec_parameters_to_context(decoder_ctx_, video->codecpar) < 0) {
    LOG(FATAL) << "Avcodec parameters copy error";
  }

  decoder_ctx_->get_format = GetHwFormat;

  if (HwDecoderInit(decoder_ctx_, this->type_) < 0)
    LOG(FATAL) << "HW decoder init error";

  if (int ret = avcodec_open2(decoder_ctx_, this->decoder_, nullptr);ret != 0) {
    char error_buf[512];
    snprintf(error_buf, 512, "Failed to open codec for stream #%u", this->video_stream_);
    LOG(FATAL) << error_buf;
  }
}

AVPixelFormat GpuDecoder::GetHwFormat(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  const AVPixelFormat *p;
  for (p = pix_fmts; *p != -1; p++) {
    if (*p == hw_pix_fmt_)
      return *p;
  }
  return AV_PIX_FMT_NONE;
}
std::optional<cv::cuda::GpuMat> GpuDecoder::ConvertFrame(AVFrame *cu_frame) {
  return this->adapter_.ConvertFrame(cu_frame);
}



