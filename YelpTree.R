library(rpart)
library(tree)
FinalrestaurantSet <- read.table("/Users/prateek/DataSets/YELP_Career/FinalDataSet.txt")
freshness <- as.numeric(abs(max(as.Date(FinalrestaurantSet$date)) - as.Date(FinalrestaurantSet$date) + 1)) # to avoid having zeros...

mydata <- data.frame(cbind(biz_rating=FinalrestaurantSet$BizRating, rev_Star=FinalrestaurantSet$ReviewStar, rev_len=FinalrestaurantSet$revLength, 
                          rev_sentiment=FinalrestaurantSet$sentiment, rev_count=FinalrestaurantSet$NumReviews, 
                           usr_Avg_Stars=FinalrestaurantSet$UsrAvgStars,
                           rev_useful=FinalrestaurantSet$rev_vote_useful, freshness, 
                           biz_longitude=FinalrestaurantSet$longitude, biz_latitude=FinalrestaurantSet$latitude, 
                           num_biz_checkin=FinalrestaurantSet$BizCheckIN
))
colnames(mydata) <- c("biz_rating", "rev_Star", "rev_len", "rev_sentiment", "rev_count", "usr_Avg_Stars", 
                      "rev_useful", "freshness", "biz_longitude", "biz_latitude", "num_biz_checkin")
smp_size <- floor(0.70 * nrow(mydata))
## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(mydata)), size = smp_size)

trainTree <- mydata[train_ind, ]
testTree <- mydata[-train_ind, ]
logistic_BizStar <- trainTree$biz_rating
logistic_BizStar[logistic_BizStar < 4] <- 0
logistic_BizStar[logistic_BizStar > 3] <- 1

logistic_BizStarTest <- testTree$biz_rating
logistic_BizStarTest[logistic_BizStarTest < 4] <- 0
logistic_BizStarTest[logistic_BizStarTest > 3] <- 1

tree_fit = rpart(logistic_BizStar~rev_Star+rev_len+rev_sentiment
            +rev_count+usr_Avg_Stars+rev_useful+freshness
            +biz_longitude+biz_latitude+num_biz_checkin, method="class", data=trainTree)

printcp(tree_fit) # display the results
plotcp(tree_fit) # visualize cross-validation results

plot(tree_fit, uniform=TRUE, main="Classification Tree for Business Ratings")
text(tree_fit, use.n=TRUE, all=TRUE, cex=.8)

fit.preds = predict(tree_fit,newdata=testTree, type="class")
fit.table = table(logistic_BizStarTest, fit.preds)
fit.table

library(ROCR)
fit.pr = predict(tree_fit,newdata=testTree,type="prob")[,2]
fit.pred = prediction(fit.pr,logistic_BizStarTest)
fit.perf = performance(fit.pred,"tpr","fpr")
plot(fit.perf,lwd=2,col="blue",
     main="ROC: Classification Trees")
abline(a=0,b=1)

tree_Biz = tree(logistic_BizStar~rev_Star+rev_len+rev_sentiment
                +rev_count+usr_Avg_Stars+rev_useful+freshness
                +biz_longitude+biz_latitude+num_biz_checkin,data=trainTree )

library(ROCR)
fit.pr = predict(tree_Biz,newdata=testTree,type="prob")[,2]
fit.pred = prediction(fit.pr,logistic_BizStarTest)
fit.perf = performance(fit.pred,"tpr","fpr")
plot(fit.perf,lwd=2,col="blue",
     main="ROC:  Classification Trees on Adult Dataset")
abline(a=0,b=1)


summary(tree_fit)
summary(tree_Biz)
plot(tree_Biz)
text(tree_Biz, all = T)

############################################################################################################
#           BOOSTING>>>>>>>
############################################################################################################
library(gbm)
boost.biz=gbm(logistic_BizStar~. -biz_rating, data=trainTree, distribution="bernoulli", n.tree=10000)
summary(boost.biz)
yhat.boost=predict(boost.biz, newdata=testTree, n.trees=10000)
mean((yhat.boost -logistic_BizStarTest)^2)

fit.pr = predict(boost.biz,newdata=testTree, n.trees=10000, type="response")
fit.pred = prediction(fit.pr,logistic_BizStarTest)
fit.perf = performance(fit.pred,"tpr","fpr")
plot(fit.perf,lwd=2,col="blue",
     main="ROC: Boosting on Yelp Dataset")
abline(a=0,b=1)

############################################################################################################
#           BAGGING>>>>>>>
############################################################################################################

library(randomForest)
bag.Biz=randomForest(logistic_BizStar~.-biz_rating, data=trainTree, distribution="bernoulli", importance=TRUE)
print(bag.Biz) # view results 
importance(bag.Biz) # importance of each predictor
summary(bag.Biz)
yhat.bag=predict(bag.Biz, newdata=testTree)
mean((yhat.bag -logistic_BizStarTest)^2)

fit.pr = predict(bag.Biz,newdata=testTree,type="response")
fit.pred = prediction(fit.pr,logistic_BizStarTest)
fit.perf = performance(fit.pred,"tpr","fpr")
plot(fit.perf,lwd=2,col="blue",
     main="ROC: RandomForest on Yelp Dataset")
abline(a=0,b=1)

############################################################################################################
#           SVM>>>>>>>
############################################################################################################

