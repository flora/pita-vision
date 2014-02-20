//
//  CameraViewController.m
//  MLPVision
//
//  Created by Flora on 2/20/14.
//  Copyright (c) 2014 Flora. All rights reserved.
//

#import "CameraViewController.h"
#import "LogoDetector.h"

@interface CameraViewController ()

@property LogoDetector *logoDetector;

@end

@implementation CameraViewController

- (id)init
{
    self = [super init];
    if (self) {
        // Custom initialization
    }
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor redColor];
	// Do any additional setup after loading the view.
    self.logoDetector = [[LogoDetector alloc] init];
    [self triggerCameraMode];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (void)triggerCameraMode
{
    if ([UIImagePickerController isSourceTypeAvailable: UIImagePickerControllerSourceTypeCamera])
    {
        UIImagePickerController *imagePicker = [[UIImagePickerController alloc] init];
        imagePicker.delegate = self;
        imagePicker.sourceType = UIImagePickerControllerSourceTypeCamera;
        imagePicker.allowsEditing = NO;
    } else {
        NSLog(@"Failed to find camera device.");
    }
}

-(void)imagePickerController:(UIImagePickerController *)picker
didFinishPickingMediaWithInfo:(NSDictionary *)info
{
    [self dismissViewControllerAnimated:YES completion:nil];
    
    // TODO: Use results of photo
    UIImage *anImage = [info valueForKey:UIImagePickerControllerOriginalImage];
    [self.logoDetector detectLogo:anImage];
}

-(void)imagePickerControllerDidCancel:(UIImagePickerController *)picker
{
    [self dismissViewControllerAnimated:YES completion:nil];
}

@end
