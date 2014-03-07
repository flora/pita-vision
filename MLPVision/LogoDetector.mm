//
//  LogoDetector.m
//  MLPVision
//
//  Created by Flora on 2/20/14.
//  Copyright (c) 2014 Flora. All rights reserved.
//

#include "LogoDetector.h"
#include <stdio.h>
#include <dirent.h>
#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

//extern "C" {
//# include "svm_common.h"
//# include "svm_learn.h"
//}

// #include "svmlight/svmlight.h"

using namespace std;
using namespace cv;

#define MAX_BUF 1048

static string basePath = "/Users/flora/CodingProjects/pita-vision/MLPVision/";
static string posSamplesDir = basePath+"images/hotpockets/train/";
static string negSamplesDir = basePath+"images/negative_train/";
static string posTestDir = basePath+"images/hotpockets/test/positive/";
static string negTestDir = basePath+"images/hotpockets/test/negative/";
// static string featuresFile = basePath+"genfiles/hotpocket_features.dat";
// static string svmModelFile = basePath+"genfiles/hotpocket_svm.dat";
// static string descriptorVectorFile = basePath+"genfiles/hotpocket_descriptor.dat";

static const cv::Size trainingPadding = cv::Size(0,0);
static const cv::Size winStride = cv::Size(8,8);

@implementation LogoDetector : NSObject

- (LogoDetector *)init {
    [self trainDetector];
    return self;
}

- (void)dealloc
{
    // TODO
}

//- (id)init
//{
//    self = [super init];
//    if (self) {
//        // Custom initialization
//    }
//    return self;
//}

- (string)toLowerCase:(const string&) in {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

- (void)storeCursor {
    printf("\033[s");
}

- (void)resetCursor {
    printf("\033[u");
}

- (void)getFilesInDirectory:(const string&)dirName fileNames:(vector<string>&)fileNames extensions:(const vector<string>&)validExtensions {
    printf("Opening directory %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    
    DIR* dp = opendir(dirName.c_str());
    
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = [self toLowerCase:(string(ep->d_name).substr(extensionLocation + 1))];
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
//                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

- (void) calculateFeaturesFromInput:(const string&) imageFilename featureVec:(vector<float>&) featureVector hog:(HOGDescriptor&) hog {

    Mat imageData = imread(imageFilename, 0);
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    Mat imageDataResized;
    resize(imageData, imageDataResized, hog.winSize, 0, 0, INTER_CUBIC);
    // Check for mismatching dimensions
    if (imageDataResized.cols != hog.winSize.width || imageDataResized.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageDataResized.cols, imageDataResized.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<cv::Point> locations;
    hog.compute(imageDataResized, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
    imageDataResized.release();
}

/**
 * Test detection with custom HOG description vector
 * @param hog
 * @param imageData
 */
- (void) detectTest:(const HOGDescriptor&) hog imData:(Mat&) imageData {
    vector<cv::Rect> found;
    int groupThreshold = 2;
    cv::Size padding(cv::Size(32, 32));
    cv::Size winStride(cv::Size(8, 8));
    double hitThreshold = 0.; // tolerance
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, 1.05, groupThreshold);
    if (found.size() > 0) {
        NSLog(@"Found a pocket\n");
    } else {
        NSLog(@"No pocket\n");
    }
    // showDetections(found, imageData);
}

- (void)trainDetector {
    
    HOGDescriptor hog;
    hog.winSize = cv::Size(64, 128);
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> positiveTestImages;
    static vector<string> negativeTestImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    [self getFilesInDirectory:posSamplesDir fileNames:positiveTrainingImages extensions:validExtensions];
    [self getFilesInDirectory:negSamplesDir fileNames:negativeTrainingImages extensions:validExtensions];
    
    int featVecSize = 3780;
    int overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    
//    float** labels = (float **) malloc(overallSamples * sizeof(float *));
//    float** trainingData = (float **) malloc (overallSamples * sizeof(float *));
//    for (int i = 0; i < overallSamples; i++)
//        trainingData[i] = (float *) malloc(featVecSize * sizeof(float));
//    for (int i = 0; i < overallSamples; i++)
//        labels[i] = (float *) malloc(sizeof(float));
//    float labels[overallSamples];
//    float trainingData[overallSamples][featVecSize];
    CvMat* labels = cvCreateMat(overallSamples, 1, CV_32FC1);
    CvMat* trainingData = cvCreateMat(overallSamples, featVecSize, CV_32FC1);
    cvZero(labels);
    cvZero(trainingData);
//    NSLog(@"labels cols: %d\n", labels->cols);
//    NSLog(@"labels rows: %d\n", labels->rows);
//    NSLog(@"trainingData cols: %d\n", trainingData->cols);
//    NSLog(@"trainingData rows: %d\n", trainingData->rows);
    
//    NSLog(@"hiiiiiiiiii");
    
//    float percent;
    for (unsigned long currentFile=0; currentFile < overallSamples; ++currentFile) {
//        [self storeCursor];
        vector<float> featureVector;
        // Get positive or negative sample image file path
//        NSLog(currentFile < positiveTrainingImages.size() ? @"true" : @"false");
        const string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
        // Output progress
//        if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
//            percent = ((currentFile+1) * 100 / overallSamples);
//            printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
//            fflush(stdout);
//            [self resetCursor];
//        }
        fflush(stdout);
        // Calculate feature vector from current image file
        [self calculateFeaturesFromInput:currentImageFile featureVec:featureVector hog:hog];
        cvmSet(labels, currentFile, 0, (currentFile < positiveTrainingImages.size()) ? 1.0 : -1.0);
//        labels[currentFile][0] = (currentFile < positiveTrainingImages.size()) ? 1.0 : -1.0;
        // TODO : save featureVector
        float *p = &featureVector[0];
        for (int i = 0; i < featVecSize; i++) {
            cvmSet(trainingData, currentFile, i, p[i]);
//            trainingData[currentFile][i] = p[i];
        }
    }
    
    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1);
    
    CvSVM* svm = new CvSVM(trainingData, labels, Mat(), Mat(), params);
    
//    CvSVM SVM;
//    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    
    [self getFilesInDirectory:posTestDir fileNames:positiveTestImages extensions:validExtensions];
    [self getFilesInDirectory:negTestDir fileNames:negativeTestImages extensions:validExtensions];
    
    unsigned long overallTestSamples = positiveTestImages.size() + negativeTestImages.size();
    
    for (unsigned long currentFile=0; currentFile < overallTestSamples; ++currentFile) {
        vector<float> featureVector;
        float classification = (currentFile < positiveTestImages.size()) ? 1.0 : -1.0;
        const string currentImageFile = (currentFile < positiveTestImages.size() ? positiveTestImages.at(currentFile) : negativeTestImages.at(currentFile - positiveTestImages.size()));
        [self calculateFeaturesFromInput:currentImageFile featureVec:featureVector hog:hog];
        CvMat* cvIm = cvCreateMat(1, featVecSize, CV_32FC1);
        float *p = &featureVector[0];
        for (int i=0; i<featVecSize; i++) {
                cvmSet(cvIm, 0, i, p[i]);
        }
        float prediction = svm->predict(cvIm);
//        NSLog(@"class: %s\n", classification);
//        NSLog(@"predict: %f\n", prediction);
        if (classification == prediction) {
            NSLog(@"%s\n", currentImageFile.c_str());
        }
    }
    
    cvReleaseMat(&labels);
	cvReleaseMat(&trainingData);
    
}

- (void)testDetector: (const string&) imageFilename svmModel: (CvSVM) svm {
    
    Mat imageData = imread(imageFilename, 0);
}

- (void)detectLogo:(UIImage*)img {
    // TODO : FILL
}

@end
