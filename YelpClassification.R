library(MASS)
FinalrestaurantSet <- read.table("/Users/prateek/DataSets/YELP_Career/FinalDataSet.txt")
freshness <- as.numeric(abs(max(as.Date(FinalrestaurantSet$date)) - as.Date(FinalrestaurantSet$date) + 1)) # to avoid having zeros...

mydata <- data.frame(cbind(biz_rating=FinalrestaurantSet$BizRating, rev_Star=FinalrestaurantSet$ReviewStar, rev_len=FinalrestaurantSet$revLength, rev_sentiment=FinalrestaurantSet$sentiment, rev_count=FinalrestaurantSet$NumReviews, 
                usr_Avg_Stars=FinalrestaurantSet$UsrAvgStars,
                rev_useful=FinalrestaurantSet$rev_vote_useful, freshness, 
                biz_longitude=FinalrestaurantSet$longitude, biz_latitude=FinalrestaurantSet$latitude, 
                num_biz_checkin=FinalrestaurantSet$BizCheckIN
))
################################################################################################
#####  LOGISTIC REGRESSION CLASSIFICATION>>>
################################################################################################

smp_size <- floor(0.70 * nrow(mydata))
## set the seed to make your partition reproductible
set.seed(999)
train_ind <- sample(seq_len(nrow(mydata)), size = smp_size)

trainData <- mydata[train_ind, ]
testData <- mydata[-train_ind, ]
logistic_BizStar <- trainData$biz_rating
logistic_BizStar[logistic_BizStar < 4] <- 0
logistic_BizStar[logistic_BizStar > 3] <- 1

logistic_BizStarTest <- testData$biz_rating
logistic_BizStarTest[logistic_BizStarTest < 4] <- 0
logistic_BizStarTest[logistic_BizStarTest > 3] <- 1


glm.fit=glm(logistic_BizStar~rev_Star+rev_len+rev_sentiment
            +rev_count+usr_Avg_Stars+rev_useful+freshness+
              biz_longitude+biz_latitude+num_biz_checkin
            ,data=trainData ,family=binomial)

qda.fit=qda(logistic_BizStar~rev_Star+rev_len+rev_sentiment
            +rev_count+usr_Avg_Stars+rev_useful+freshness+
              biz_longitude+biz_latitude+num_biz_checkin
            ,data=trainData ,family=binomial)
summary(glm.fit)
plot(glm.fit)

glm.probs=predict(glm.fit,testData , type="response")
glm.pred=rep(0,length(glm.probs))
glm.pred[glm.probs >.5] <- 1
table(glm.pred, logistic_BizStarTest)

library(ROCR)
fit.pr = predict(glm.fit,newdata=testData,type="response")
fit.pred = prediction(fit.pr,logistic_BizStarTest)
fit.perf = performance(fit.pred,"tpr","fpr")
plot(fit.perf,lwd=2,col="blue",
     main="ROC: Logistic Regression")
abline(a=0,b=1)

qda.class=predict(qda.fit,testData)$class
table(qda.class ,logistic_BizStarTest)


#glm.pred    0    1
#       0  394  856
#       1 1722 3897
