Classification I
================
Lin Yang

``` r
library(caret)
library(glmnet)
library(mlbench)
library(pROC)
library(pdp)
library(vip)
library(AppliedPredictiveModeling)
```

## Dataset

``` r
data(PimaIndiansDiabetes2)
dat <- na.omit(PimaIndiansDiabetes2)

theme1 <- transparentTheme(trans = 0.4)
trellis.par.set(theme1)

featurePlot(x = dat[, 1:8],
            y = dat$diabetes,
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free")),
            plot = "density", pch = "|",
            auto.key = list(columns = 2))
```

![](p8106_classification_I_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

## Divide data into training and test sets

``` r
set.seed(1)
rowTrain <- createDataPartition(y = dat$diabetes,
                                p = 0.75,
                                list = FALSE)
```

## Logistic regression and its cousins

### use `glm`

``` r
contrasts(dat$diabetes)
```

    ##     pos
    ## neg   0
    ## pos   1

``` r
glm.fit <- glm(diabetes ~ .,
               data = dat,
               subset = rowTrain,
               family = binomial(link = "logit"))

summary(glm.fit)
```

    ## 
    ## Call:
    ## glm(formula = diabetes ~ ., family = binomial(link = "logit"), 
    ##     data = dat, subset = rowTrain)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.8426  -0.7118  -0.3970   0.6963   2.4158  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -8.6375872  1.3353227  -6.469 9.90e-11 ***
    ## pregnant     0.1062499  0.0636546   1.669   0.0951 .  
    ## glucose      0.0355063  0.0065385   5.430 5.63e-08 ***
    ## pressure    -0.0086717  0.0130557  -0.664   0.5066    
    ## triceps      0.0203593  0.0184871   1.101   0.2708    
    ## insulin     -0.0003983  0.0014435  -0.276   0.7826    
    ## mass         0.0507929  0.0311642   1.630   0.1031    
    ## pedigree     1.1693193  0.4627894   2.527   0.0115 *  
    ## age          0.0245899  0.0215884   1.139   0.2547    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 375.08  on 294  degrees of freedom
    ## Residual deviance: 270.87  on 286  degrees of freedom
    ## AIC: 288.87
    ## 
    ## Number of Fisher Scoring iterations: 5

### 2X2 contingency table

``` r
test.pred.prob <- predict(glm.fit, newdata = dat[-rowTrain,],
                          type = "response")
test.pred <- rep("neg", length(test.pred.prob))
test.pred[test.pred.prob > 0.5] <- "pos"
test.pred
```

    ##  [1] "pos" "neg" "neg" "pos" "neg" "neg" "neg" "neg" "pos" "neg" "neg" "neg"
    ## [13] "neg" "neg" "neg" "pos" "pos" "neg" "pos" "neg" "neg" "neg" "neg" "pos"
    ## [25] "neg" "neg" "neg" "neg" "pos" "pos" "neg" "neg" "pos" "neg" "pos" "neg"
    ## [37] "neg" "neg" "neg" "neg" "neg" "neg" "neg" "neg" "neg" "pos" "pos" "neg"
    ## [49] "neg" "neg" "neg" "neg" "neg" "neg" "neg" "pos" "neg" "pos" "neg" "neg"
    ## [61] "pos" "pos" "neg" "pos" "neg" "pos" "pos" "neg" "neg" "pos" "neg" "pos"
    ## [73] "neg" "neg" "neg" "pos" "neg" "neg" "neg" "neg" "neg" "pos" "neg" "neg"
    ## [85] "neg" "neg" "pos" "neg" "neg" "neg" "neg" "neg" "pos" "neg" "neg" "neg"
    ## [97] "neg"

``` r
confusionMatrix(data = as.factor(test.pred),
                reference = dat$diabetes[-rowTrain],
                positive = "pos")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg  60  11
    ##        pos   5  21
    ##                                          
    ##                Accuracy : 0.8351         
    ##                  95% CI : (0.746, 0.9027)
    ##     No Information Rate : 0.6701         
    ##     P-Value [Acc > NIR] : 0.0002081      
    ##                                          
    ##                   Kappa : 0.6083         
    ##                                          
    ##  Mcnemar's Test P-Value : 0.2112995      
    ##                                          
    ##             Sensitivity : 0.6562         
    ##             Specificity : 0.9231         
    ##          Pos Pred Value : 0.8077         
    ##          Neg Pred Value : 0.8451         
    ##              Prevalence : 0.3299         
    ##          Detection Rate : 0.2165         
    ##    Detection Prevalence : 0.2680         
    ##       Balanced Accuracy : 0.7897         
    ##                                          
    ##        'Positive' Class : pos            
    ## 

### ROC curve

``` r
roc.glm <- roc(dat$diabetes[-rowTrain], test.pred.prob)
plot(roc.glm, legacy.axes = TRUE, print.auc = TRUE)
plot(smooth(roc.glm), col = 4, add = TRUE)
```

![](p8106_classification_I_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

### Use `caret` to compare CV performance with other models

``` r
ctrl <- trainControl(method = "repeatedcv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)
set.seed(1)
model.glm <- train(x = dat[rowTrain, 1:8],
                   y = dat$diabetes[rowTrain],
                   method = "glm",
                   metric = "ROC",
                   trControl = ctrl)
```
