### What determines the principle componenets?

#### The principle components are determined from the eigenvectors obtained from the eigenvalue decomposition, which are orthogonal directions that capture the maximum variance in the data. 
#### In the SVD, the left singular vectors correspond to the eigen vectors of the covariance matrix.


### What determines the positive or negative correlations between the components?

#### To calaulate the correlation from the covariance matrix, we use the formula cor(x,y)= cov(x,y)/ sqrt(var(x) * var(y))
#### cor(x,y)= 2/ sqrt(4*3)= 0.5774
#### cor(x,z)= 1/ sqrt(4*2)= 0.3536
#### cor(y,z)= 1.5/ sqrt(3*2)= 0.6124
#### Correlation matrix: [[1, 0.5774, 0.3536] [0.5774, 1, 0.6124] [0.3536, 0.6124, 1]]
#### All three of the variables have a positive correlation, but they do not have a very strong correlation. The correlation cor(x,y) are considered moderate, the correlation cor(x,z) are relatively weak, and the correlation cor(y,z) are also moderate.

### Does the projection agree with the poitive or negative correlations of the original matrix.

#### No, because the signs of the first two principal components does not match the signs of the correlations.