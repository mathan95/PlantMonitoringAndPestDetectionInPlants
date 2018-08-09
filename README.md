# PlantMonitoringAndPestDetectionInPlants
This is one of my project as an intern where I hav to monitor the plants and detect when there is a pest. 

This contains codes for the following function-
            Images are acquired and background removed images of plant leaves are given as an input. Then the algorithms 
            wiill do the necessay pre processing and will localize any important things in leaves which it consider as 
            alien to the image and then those will be segmented out and each will be given to a classifier which will
            check each images and detect which one of them are actual pests and will only output the image of the pest.
            
            This part is actually to be implemented so as the classifier classifying the pest with it is name.
            For that I am going to finetune the vgg19 classifier with the pest data available. But in this implementation
            this is not found. Only the classifier will give whether there is a pest or not.
            
 The files in the directory-
            combined_pest_detection-
                    1.the algorithm here use the green removal algorithm to remove the leave parts out of the picture
                    2.then it removes the noises found in the image
                    3.then morphological operations are carried out to the grayscale image to correctly localize the pests
                    4.after it connected component analysis is used to find the remaining components and then they are localized
                      by localization algorithm
                    5.after that vgg19 classifier is used find the pest part and then it is revealed
                    
            combined_pest_detection_with_clustering
                    only difference in this implementation is that in place of the first step in the earlier algorithm, here
                    clustering is used first and green removal is only used with the cluster centers.
                    
          
