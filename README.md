# face-mosaic-with-real-time-patch

Facial detection and recognition technology is widely used in identity verification and security applications, such as calculating foot traffic, automated airport clearance systems, access control systems, and smart mobile devices.

In terms of personal privacy, it is an area that many people are deeply concerned about. For example, during street interviews or live broadcasts, individuals who do not wish to appear on camera often prefer to have their images pixelated or blurred to protect their rights to their own likeness.

In today's mosaic processing methods, it is typically necessary to annotate and process human faces  in post-production after the video has been recorded.

However, our research motivation and objective are to instantly identify specific individuals during the filming process and apply privacy protection to them without the need for cumbersome post-processing steps. 

We use MTCNN(https://github.com/timesler/facenet-pytorch) as the face recognition model and a two colors patch as the basis for judgment whether or to blur the face

Assuming red and blue are used as triggers, we store the positions of pixels that fall within the color ranges of these two colors.

We shift the position of the red pixels and check for overlaps with the position of the blue pixels, then store the positions of the overlapping pixels.

Based on these pixels, we calculate their positions relative to the central point.

![image](https://github.com/B10732006/face-mosaic-with-real-time-patch/assets/71718508/5b280558-dc46-4aa3-848f-092463988ea9)












