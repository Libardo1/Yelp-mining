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

trainSVM <- mydata[train_ind, ]
testSVM <- mydata[-train_ind, ]
classified_BizStar <- trainSVM$biz_rating
classified_BizStar[classified_BizStar < 4] <- 0
classified_BizStar[classified_BizStar > 3] <- 1

classified_BizStarTest <- testSVM$biz_rating
classified_BizStarTest[classified_BizStarTest < 4] <- 0
classified_BizStarTest[classified_BizStarTest > 3] <- 1

############################################################################################################
#           SVM>>>>>>>
############################################################################################################

library(e1071)
svm.fit=svm(logistic_BizStar~. -biz_rating, data=trainSVM, kernel="radial", cost=10, scle=FALSE)
summary(svm.fit)
ypred=predict(svm.fit ,newdata=testSVM)
mean((ypred -classified_BizStarTest)^2)
table(ypred,classified_BizStarTest )

fit.pr = predict(svm.fit,newdata=testSVM ,type="response")
fit.pred = prediction(fit.pr,classified_BizStarTest)
fit.perf = performance(fit.pred,"tpr","fpr")
plot(fit.perf,lwd=2,col="blue",
     main="ROC: SVM on Yelp Dataset")
abline(a=0,b=1)

############################################################################################################
#           OtherStuff>>>>>>>
############################################################################################################

pr.out=prcomp(trainSVM, scale=TRUE)
summary(pr.out)
biplot(pr.out, scale=0)


km.out=kmeans(trainSVM,2,nstart=20)
plot(trainSVM, col=(km.out$cluster +1), main="K-Means Clustering Results with K=2", pch=20, cex=2)

hc.single=hclust(dist(trainSVM), method="single")
plot(hc.single,main="Complete Linkage", cex =.9)
