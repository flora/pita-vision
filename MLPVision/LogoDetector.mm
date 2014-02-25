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
#include <mach-o/dyld.h>	/* _NSGetExecutablePath */

// #include "svmlight/svmlight.h"

using namespace std;
using namespace cv;

#define MAX_BUF 1048

static string basePath = "/Users/flora/CodingProjects/pita-vision/MLPVision/";
static string posSamplesDir = "images/hotpockets/train/";
static string negSamplesDir = "images/negative_train/";
static string featuresFile = "genfiles/hotpocket_features.dat";
static string svmModelFile = "genfiles/hotpocket_svm.dat";
static string descriptorVectorFile = "genfiles/hotpocket_descriptor.dat";

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
                printf("Found matching data file '%s'\n", ep->d_name);
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

- (void)trainDetector {
    
    HOGDescriptor hog;
    hog.winSize = cv::Size(64, 128);
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    [self getFilesInDirectory:basePath+posSamplesDir fileNames:positiveTrainingImages extensions:validExtensions];
    [self getFilesInDirectory:basePath+negSamplesDir fileNames:negativeTrainingImages extensions:validExtensions];
    
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    NSLog(@"num samples: %lu\n", overallSamples);
    
    
    
}

- (void)detectLogo:(UIImage*)img {
    // TODO : FILL
}

@end
