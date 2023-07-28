#ifndef IMAGE_TOOLS_IMG_TOOL_H
#define IMAGE_TOOLS_IMG_TOOL_H

#include "iostream"
#include "string"
#include "filesystem"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

cv::Mat _readTiffImage(const std::string& filename, bool forceProPhotoRGB);
cv::Mat _ImageSquarePadding(cv::Mat inputImage, float borderRatio);
cv::Mat _imageColorSpaceConvert(cv::Mat inputImage, int target);
void _saveJPG(cv::Mat inputImage, const std::string& out_filename, int compression);
cv::Mat _processImage(const std::string& filename);
void processImages(const std::string& inputDir, const std::string& outputDir);

#endif //IMAGE_TOOLS_IMG_TOOL_H
