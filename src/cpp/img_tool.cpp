#include "img_tool.h"


cv::Mat _readTiffImage(const std::string& filename, bool forceProPhotoRGB) {
    cv::Mat image;

    image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Failed to read image " << filename << std::endl;
        return image;
    }
    if (forceProPhotoRGB && (image.type() != CV_16UC3)) {
        std::cerr << "Image does not have 16-bit color depth or the number of channels is not 3!" << std::endl;
    }
    return image;
}


cv::Mat _ImageSquarePadding(cv::Mat inputImage, float borderRatio = 0.) {
    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;

    int targetSize = std::max(inputWidth, inputHeight) * (1 + 2 * borderRatio);
    cv::Mat targetImage(targetSize, targetSize, CV_16UC3, cv::Scalar(255, 255, 255));

    int offsetX = (targetSize - inputWidth) / 2;
    int offsetY = (targetSize - inputHeight) / 2;

    inputImage.copyTo(targetImage(cv::Rect(offsetX, offsetY, inputWidth, inputHeight)));

    return targetImage;
}

cv::Mat _imageColorSpaceConvert(cv::Mat inputImage, int target) {
    cv::Mat targetImage;
    if (target != CV_16UC3) {
        cv::normalize(inputImage, targetImage, 0, 255, cv::NORM_MINMAX, target);
    } else {
        targetImage = inputImage.clone();
    }
    return targetImage;
}


void _saveJPG(cv::Mat inputImage, const std::string& out_filename, int compression = 100) {
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(compression);
    cv::imwrite(out_filename, inputImage, compression_params);
}

cv::Mat _processImage(const std::string& inputPath, const std::string& outputPath, const float borderRatio = 0.) {
    cv::Mat img = _readTiffImage(inputPath, true);
    img = _imageColorSpaceConvert(img, CV_8U);
    img = _ImageSquarePadding(img, borderRatio);
    _saveJPG(img, outputPath, 90);
    return img;
}

void processImages(const std::string& inputDir, const std::string& outputDir, const float borderRatio) {
    for (const auto& entry: std::filesystem::directory_iterator(inputDir)) {
        const std::string inPath = entry.path().string();

        if (inPath.ends_with(".tif") || inPath.ends_with(".tiff")) {
            std::cout << "Processing image " << inPath << std::endl;

            const std::string outPath = outputDir + "/" + entry.path().filename().replace_extension(".jpg").string();

            _processImage(inPath, outPath, borderRatio);

//            std::cout << "Output at " << outPath << std::endl;
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <source_directory_path> <target_directory_path> <borderRatio>" << std::endl;
        return 1;
    }
    const std::string inputDir = argv[1];
    const std::string outputDir = argv[2];
    const float borderRatio = std::stof(argv[3]);

    processImages(inputDir, outputDir, borderRatio);
}