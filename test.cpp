#include <cstdio>
//opencv
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include "opencv2/opencv.hpp"
//cuda
#include "cuda_runtime_api.h"
#include "help_cuda.h"
//glog
#include "glog/logging.h"
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

static FILE *output_file = nullptr;
int frame_index = 0;
GpuDecoder gpu_decoder;

static int decode_write(AVCodecContext *avctx, AVPacket *packet) {
  AVFrame *frame = NULL, *sw_frame = NULL;
  AVFrame *tmp_frame = NULL;
  uint8_t *buffer = NULL;
  int size;
  int ret = 0;

  ret = avcodec_send_packet(avctx, packet);
  if (ret < 0) {
    fprintf(stderr, "Error during decoding\n");
    return ret;
  }
  while (1) {
    if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
      fprintf(stderr, "Can not alloc frame\n");
      ret = AVERROR(ENOMEM);
//      goto fail;
    }

    ret = avcodec_receive_frame(avctx, frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&frame);
      av_frame_free(&sw_frame);
      return 0;
    } else if (ret < 0) {
      fprintf(stderr, "Error while decoding\n");
//      goto fail;
    }

    if (frame->format == GpuDecoder::hw_pix_fmt_) {
      /* retrieve data from GPU to CPU */
      if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
        fprintf(stderr, "Error transferring the data to system memory\n");
//        goto fail;
      }
      tmp_frame = sw_frame;
    } else {
      tmp_frame = frame;
    }

//    cv::Mat image = avframe_to_mat(tmp_frame, frame, FrameMode::GPU);
    auto cu_image_opt = gpu_decoder.ConvertFrame(frame);
    auto cu_image = *cu_image_opt;
    cv::Mat output_image;
    cu_image.download(output_image);
    cv::cvtColor(output_image, output_image, cv::COLOR_YUV2BGR_NV12);
    cv::imwrite("frame.jpg", output_image);
    frame_index += 1;
    if (frame_index >= 15) {
      break;
    }
    fail:
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_freep(&buffer);
    if (ret < 0)
      return ret;
  }
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log";
  AVFormatContext *input_ctx = nullptr;
  int video_stream, ret;
  AVStream *video = nullptr;
  AVCodecContext *decoder_ctx = nullptr;
  const AVCodec *decoder = nullptr;
  AVPacket *packet = nullptr;
  enum AVHWDeviceType type;

  if (argc < 4) {
    fprintf(stderr, "Usage: %s <device type> <input file> <output file>\n", argv[0]);
    return -1;
  }

  type = av_hwdevice_find_type_by_name(argv[1]);
  if (type == AV_HWDEVICE_TYPE_NONE) {
    fprintf(stderr, "Device type %s is not supported.\n", argv[1]);
    fprintf(stderr, "Available device types:");
    while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
      fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
    fprintf(stderr, "\n");
    return -1;
  }

  packet = av_packet_alloc();
  if (!packet) {
    fprintf(stderr, "Failed to allocate AVPacket\n");
    return -1;
  }

  if (avformat_open_input(&input_ctx, argv[2], nullptr, nullptr) != 0) {
    fprintf(stderr, "Cannot open input file '%s'\n", argv[2]);
    return -1;
  }

//  GpuDecoder gpu_decoder;
  gpu_decoder.Init(input_ctx);
  decoder_ctx = gpu_decoder.GetAVCodecContext();
  video_stream = gpu_decoder.GetVideoStream();

  output_file = fopen(argv[3], "w+b");

  while (ret >= 0) {
    if ((ret = av_read_frame(input_ctx, packet)) < 0)
      break;

    if (video_stream == packet->stream_index)
      ret = decode_write(decoder_ctx, packet);

    av_packet_unref(packet);
  }

  ret = decode_write(decoder_ctx, nullptr);

  if (output_file)
    fclose(output_file);
  av_packet_free(&packet);

  return 0;
}
