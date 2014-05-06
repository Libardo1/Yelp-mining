## based in http://mason.gmu.edu/~csutton/tactR789cr.txt

##

# #install.packages("class")
# #install.packages("MASS")
# install.packages("Hmisc")
# install.packages("classPP")
# install.packages("klaR")
# #install.packages("e1071") 
# install.packages("kknn")
# #install.packages("rpart")
# install.packages("boost")
# install.packages("mvtnorm")          
# install.packages("multinomRob")   
# install.packages("lars")
# #install.packages("stats")
# install.packages("leaps")

# library(class)
# library(MASS)
# library(Hmisc)
# library(classPP)
# library(klaR)
# library(e1071) 
# library(kknn)
# library(rpart)
# library(boost)
# library(mvtnorm)          
# library(multinomRob )   
# library(lars)
# library(stats)
# library(leaps)

################################################################################################
#####     LASSO REGRESSION
################################################################################################
x_train <- cbind(trainDataS$rev_Star, trainDataS$rev_len, trainDataS$rev_sentiment, trainDataS$rev_count,
                 trainDataS$usr_Avg_Stars, trainDataS$rev_useful, trainDataS$freshness,
                 trainDataS$biz_longitude, trainDataS$biz_latitude, trainDataS$num_biz_checkin)

x_test <- cbind(testDataS$rev_Star, testDataS$rev_len, testDataS$rev_sentiment, testDataS$rev_count,
                 testDataS$usr_Avg_Stars, testDataS$rev_useful, testDataS$freshness,
                 testDataS$biz_longitude, testDataS$biz_latitude, testDataS$num_biz_checkin)

model_laso <- lars(x_train,trainDataS$biz_rating,type="lasso")
model_laso
plot(model_laso,plottype="coefficients")
plot(model_laso, plottype="Cp")

# By using cross-validation with the lasso,
# a good (hopefully near-optimal) value for
# the "fraction" can be determined.
cvlas <- cv.lars(x_train,trainDataS$biz_rating, type="lasso")
cvlas
frac <- cvlas$fraction[which.min(cvlas$cv)]
frac
model_laso.coef <- predict.lars(model_laso, type="coefficients", mode="fraction", s=frac)
model_laso.coef
#yhat_rt <- predict(model_laso,mydata)

# As a check, let's see if setting the value of
# s (the fraction, in the mode being used) to 1
# yields the coefficient values from the OLS fit.
model_laso.coef <- predict.lars(model_laso, type="coefficients", mode="fraction", s=1)
model_laso.coef
model_reg

yhat_lass <- predict.lars(model_laso, x_train, type="fit")
mse_lass<-MSE_WM(yhat_lass$fit[,1],trainDataS$biz_rating)
mse_lass

yhatVal_lass <- predict.lars(model_laso, x_test, type="fit")
mseVal_lass<-MSE_WM(yhatVal_lass$fit[,1],testDataS$biz_rating)
mseVal_lass

x<-seq(0,2927)
df<-data.frame(x,testDataS$biz_rating,yhatVal_lass$fit[,1])
df2<-melt(data=df,id.vars="x")
ggplot(data=df2, aes(x=x, y=value,colour=variable))+geom_point()

################################################################################################
#####     RIDGE REGRESSION
################################################################################################

#model_ridge <- lm.ridge(rev_stars~rev_length+star_diff+freshness+useful_vote+funny_vote+cool_vote+stars_buz+reviewCount_buz+AvgStarUsr, lambda = seq(0, 10, 1))
model_ridge <- lm.ridge(trainDataS$biz_rating~.,trainDataS, lambda = seq(0, 10, 1))
model_ridge$kHKB
model_ridge$kLW
model_ridge$GCV

# Two estimates of (a good) lambda are
# about 2.7 and 4.0, and generalized
# cross-validation suggests using a 
# value between 6 and 8.  <I'll skip
# showing some of the work I did to
# narrow in on the interval (6.72, 6.84).>
model_ridge <- lm.ridge(trainDataS$biz_rating~.,trainDataS, lambda = seq(6.72, 6.84, 0.01))
model_ridge$GCV


#
# After some searching, I determined that 
# the value of lambda which minimizes the
# generalized cross-validation estimate of
# the error is about 6.8.  Two estimation
# methods produced estimated lambdas of 
# about 2.7 and 4.0.  So I will fit three
# ridge regression models, using lambdas 
# values of 2.7, 4.0, and 6.8.  As a check,
# I will fit a fourth model using 0 for lambda,
# which should be the same as OLS.
model_ridge1 <- lm.ridge(trainDataS$biz_rating~.,trainDataS, lambda = 0.7)
#ridge1 <- lm.ridge(y ~ sx1 + sx2 + sx3 + sx4 + sx5 + sx6, lambda = 2.7)
model_ridge2 <- lm.ridge(trainDataS$biz_rating~.,trainDataS, lambda = 4.0)
#ridge2 <- lm.ridge(y ~ sx1 + sx2 + sx3 + sx4 + sx5 + sx6, lambda = 4.0)
model_ridge3 <- lm.ridge(trainDataS$biz_rating~.,trainDataS, lambda = 6.8)
#ridge3 <- lm.ridge(y ~ sx1 + sx2 + sx3 + sx4 + sx5 + sx6, lambda = 6.8)
model_ridge2 <- lm.ridge(trainDataS$biz_rating~.,trainDataS, lambda = 0)
#ridge4 <- lm.ridge(y ~ sx1 + sx2 + sx3 + sx4 + sx5 + sx6, lambda = 0)

yhatVal_lass <- predict(model_ridge)
yhatVal.ridge = scale(testDataS[,2:11],center = F, scale = model_ridge$scales)%*% model_ridge$coef[,which.min(model_ridge$GCV)] + model_ridge$ym

mseVal_ridge<-MSE_WM(yhatVal.ridge,testDataS$biz_rating)
mseVal_ridge

df<-data.frame(x,testDataS$biz_rating,yhatVal_lass$fit[,1])
df2<-melt(data=df,id.vars="x")
ggplot(data=df2, aes(x=x, y=value,colour=variable))+geom_point()
################################################################################################
#####     PRINCIPAL COMPONENTS REGRESSION
################################################################################################
## freshness was eliminated because PCA applies only to numerical variables
datatr.pca <- princomp(~rev_Star+rev_len+useful_vote+funny_vote+cool_vote+stars_buz+reviewCount_buz+AvgStarUsr)

datatr.pca <- princomp(~trainDataS$rev_Star+trainDataS$rev_len+trainDataS$rev_sentiment+trainDataS$rev_count+
trainDataS$usr_Avg_Stars+trainDataS$rev_useful+trainDataS$freshness+
trainDataS$biz_longitude+trainDataS$biz_latitude+trainDataS$num_biz_checkin)

summary(datatr.pca)
plot(datatr.pca)
loadings(datatr.pca)

# The next step is to compute the p.c. variable
# values from the training data.
prcomp.tr <- predict(datatr.pca)
prcomp.tr

# Now I will "pick off" the p.c. variable vectors.
# (Note: One wouldn't have to do it this way.)
pc1 <- prcomp.tr[,1]
pc2 <- prcomp.tr[,2]
pc3 <- prcomp.tr[,3]
pc4 <- prcomp.tr[,4]
pc5 <- prcomp.tr[,5]
pc6 <- prcomp.tr[,6]
pc7 <- prcomp.tr[,7]
pc8 <- prcomp.tr[,8]
pc9 <- prcomp.tr[,9]
pc10 <- prcomp.tr[,10]


# And now I will do the regressions.
# I'll start by using all of the p.c's,
# which should give me a fit equivalent 
# to the OLS fit based on all of the var's.
model_pcr <- lm(trainDataS$biz_rating ~ pc1 + pc2 + pc3 + pc4 + pc5 + pc6 + pc7+pc8+pc9+pc10)
summary(model_pcr)
mydata_pcr <- data.frame(trainDataS$biz_rating, pc1, pc2, pc3, pc4, pc5, pc6, pc7+pc8+pc9+pc10)

yhat_pcr<-predict(model_pcr,mydata_pcr)
msetrain_pcr<-MSE_WM(yhat_pcr, trainDataS$biz_rating)
msetrain_pcr

x1<-seq(0,6831)
df<-data.frame(x1,trainDataS$biz_rating,yhat_pcr)
df2<-melt(data=df,id.vars="x1")
ggplot(data=df2, aes(x=x1, y=value,colour=variable))+geom_point()

################################################################################################
#####     REGRESSION TREE
################################################################################################
#library("rpart")
model_rt <- rpart(trainDataS$biz_rating~ .,data=trainDataS,control=rpart.control(xval=5,cp=0.0001),method="anova")
#print table for diferents cp and select a cp to prune
printcp(model_rt)
library(rpart.plot)
prp(model_rt,type=1)

#library(ggplot2)
#dfplot = rbind(data.frame(Model="RT",Data="Train",Target=mydata$rev_stars, Pred=prediced))
#viewing the table we select 8 splits and cp=0.00751918385
model_rt <- prune(model_rt, cp=0.00751918385)
prp(model_rt,type=1)
yhat_rt <- predict(model_rt,testDataS)
mse_rt <- MSE_WM(yhat_rt,testDataS$biz_rating)
mse_rt

df<-data.frame(x,testDataS$biz_rating,yhat_rt)
df2<-melt(data=df,id.vars="x")
ggplot(data=df2, aes(x=x, y=value,colour=variable))+geom_point()

