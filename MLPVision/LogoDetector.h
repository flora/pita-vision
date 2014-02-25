//
//  LogoDetector.h
//  MLPVision
//
//  Created by Flora on 2/20/14.
//  Copyright (c) 2014 Flora. All rights reserved.
//

#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/objdetect/objdetect.hpp>

@interface LogoDetector : NSObject {

    CvHaarClassifierCascade *_cascade;
    CvMemStorage *_storage;
    
}

- (void)detectLogo:(UIImage*)img;

@end
