#source("/Users/prateek/DataSets/YELP_Career/YelpUtils.R")
## MSE function using model
MSE <- function(model, bdata) {
  yhat <- predict(model, bdata) 
  return(mean((yhat - bdata$biz_rating)^2) )
}

## MSE function wirhout model, it use only data 
MSE_WM <- function(yhat, bdata) {
  return(mean((yhat - bdata)^2) )
}

FinalrestaurantSet<-read.table('/Users/JavierI/Downloads/latestrfiles___/FinalDataSet.txt')
freshness <- as.numeric(abs(max(as.Date(FinalrestaurantSet$date)) - as.Date(FinalrestaurantSet$date) + 1)) # to avoid having zeros...

mydata <- data.frame(cbind(biz_rating=FinalrestaurantSet$BizRating,
                           rev_Star=FinalrestaurantSet$ReviewStar,
                           rev_len=FinalrestaurantSet$revLength,
                           rev_sentiment=FinalrestaurantSet$sentiment,
                           rev_count=FinalrestaurantSet$NumReviews, 
                           usr_Avg_Stars=FinalrestaurantSet$UsrAvgStars,
                           rev_useful=FinalrestaurantSet$rev_vote_useful, freshness, 
                           biz_longitude=FinalrestaurantSet$longitude,
                           biz_latitude=FinalrestaurantSet$latitude, 
                           num_biz_checkin=FinalrestaurantSet$BizCheckIN
))

#Scaling data to obtain a data with means=0 and SD=1
x1<-as.numeric(scale(FinalrestaurantSet$BizRating))
x2<-as.numeric(scale(FinalrestaurantSet$ReviewStar))
x3<-as.numeric(scale(FinalrestaurantSet$revLength))
x4<-as.numeric(scale(FinalrestaurantSet$sentiment))
x5<-as.numeric(scale(FinalrestaurantSet$NumReviews)) 
x6<-as.numeric(scale(FinalrestaurantSet$UsrAvgStars))
x7<-as.numeric(scale(FinalrestaurantSet$rev_vote_useful))
x8<-as.numeric(scale(freshness))
x9<-as.numeric(scale(FinalrestaurantSet$longitude))
x10<-as.numeric(scale(FinalrestaurantSet$latitude))
x11<-as.numeric(scale(FinalrestaurantSet$BizCheckIN))
x12<-as.numeric(scale(FinalrestaurantSet$UsrVotesFunny))
x13<-as.numeric(scale(FinalrestaurantSet$UsrVotesUseful))
x14<-as.numeric(scale(FinalrestaurantSet$UsrVotesCool))
x15<-as.numeric(scale(FinalrestaurantSet$rev_vote_funny))
x16<-as.numeric(scale(FinalrestaurantSet$rev_vote_cool))
x17<-as.numeric(scale(FinalrestaurantSet$NumBizReviews))
x18<-as.numeric(scale(FinalrestaurantSet$BizCheckIN))
x19<-as.numeric(scale(FinalrestaurantSet$Prob))
x20<-as.numeric(scale(FinalrestaurantSet$Stratum))
                            
#take only data without correlation
mydataS <- data.frame(cbind(biz_rating=x1, rev_Star=x2, rev_len=x3, rev_sentiment=x4,
                            rev_count=x5, usr_Avg_Stars=x6, rev_useful=x7,
                            freshness=x8, biz_longitude=x9, biz_latitude=x10, num_biz_checkin=x11))


# mydataS <- data.frame(cbind(biz_rating=x1, rev_Star=x2, rev_len=x3, rev_sentiment=x4,
#                             rev_count=x5, usr_Avg_Stars=x6, rev_useful=x7, freshness=x8, 
#                             biz_longitude=x9, biz_latitude=x10, num_biz_checkin=x11,
#                             usrVFunny=x12, usrvUsefull=x13, usrvCool=x14,revvFunny=x15,
#                             revvCool=x16,NumBizRw=x17,BizChk=x18,prob=x19,stratum=x20))

# mydataS <- data.frame(cbind(biz_rating=scale(FinalrestaurantSet$BizRating),
#                             rev_Star=scale(FinalrestaurantSet$ReviewStar),
#                             rev_len=scale(FinalrestaurantSet$revLength),
#                             rev_sentiment=scale(FinalrestaurantSet$sentiment),
#                             rev_count=scale(FinalrestaurantSet$NumReviews), 
#                             usr_Avg_Stars=scale(FinalrestaurantSet$UsrAvgStars),
#                             rev_useful=scale(FinalrestaurantSet$rev_vote_useful),
#                             scale(freshness), 
#                             biz_longitude=scale(FinalrestaurantSet$longitude),
#                             biz_latitude=scale(FinalrestaurantSet$latitude), 
#                             num_biz_checkin=scale(FinalrestaurantSet$BizCheckIN)
# ))

# mydataS <- data.frame(cbind(biz_rating=FinalrestaurantSet$BizRating,
#                             rev_Star=FinalrestaurantSet$ReviewStar,
#                             rev_len=FinalrestaurantSet$revLength,
#                             rev_sentiment=FinalrestaurantSet$sentiment,
#                             rev_count=FinalrestaurantSet$NumReviews,
#                             usr_Avg_Stars=FinalrestaurantSet$UsrAvgStars,
#                             rev_useful=FinalrestaurantSet$rev_vote_useful,
#                             freshness, biz_longitude=FinalrestaurantSet$longitude,
#                             biz_latitude=FinalrestaurantSet$latitude,
#                             num_biz_checkin=FinalrestaurantSet$BizCheckIN))

smp_size <- floor(0.70 * nrow(mydataS))
## set the seed to make your partition reproductible
set.seed(123)
# train_ind <- sample(seq_len(nrow(mydata)), size = smp_size)
# 
# trainData <- mydata[train_ind, ]
# testData <- mydata[-train_ind, ]

train_indS <- sample(seq_len(nrow(mydataS)), size = smp_size)

trainDataS <- mydataS[train_indS, ]
testDataS <- mydataS[-train_indS, ]

#########################FEATURE SET DESIGN########################################################
###################################################################################################
# AvgStarUsr...Average star by user [across multiple reviews]...
###################################################################################################
#AvgStarUsr <- matrix(0,length(rev_data$user_id),1)
######## COmmenting for now, since it'll be more complicated to use it now..will have to check back later..
# AvgStarUsr <- funny_vote*0
# i<-1
# j<-1
# 
# for (i in 1:length(FinalrestarantSet$user_id)) {
#   for (j in 1:length(usr_data$user_id)) {
#     if (rev_data$user_id[i]==usr_data$user_id[j]) {
#       AvgStarUsr[i] <- usr_data$average_stars[j]
#     }
#   }
# }

#standarization of data between 0 - 5
# rev_length_n <- rev_length*5/max(rev_length)
#freshness_n <- (freshness)*5/max(freshness)

##Whole Train Data
#mydata <- data.frame(rev_stars,rev_length,star_diff,freshness,useful_vote,funny_vote,cool_vote,stars_buz,reviewCount_buz,AvgStarUsr)



# Multiple Linear Regression
model_regS <- lm(biz_rating~ rev_Star+rev_len+rev_sentiment
                 +rev_count+usr_Avg_Stars+rev_useful+freshness+
                   biz_longitude+biz_latitude+num_biz_checkin,data=trainDataS)
 
# model_regS <- lm(biz_rating~ .,data=trainDataS)

# model_reg <- lm(stars$BizRating~ train$ReviewStar+NumReviews+stars.UserAvgStars
#                 +freshness_n+FinalrestaurntSet.rev_vote_funny
#                 +FinalrestaurntSet.rev_vote_useful+FinalrestaurntSet.rev_vote_cool
#                 ,data=train)
# 

testPredicted <- predict(model_regS, testDataS, interval ="confidence")

plot(predict(model_regS), residuals(model_regS))

# # Multiple Linear Regression
# model_reg <- lm(FinalrestarantSet$ReviewStar~FinalrestarantSet$NumReviews + FinalrestarantSet$BizRating 
#                 + FinalrestarantSet$UsrAvgStars + freshness_n 
#                 + FinalrestarantSet$UsrVotesUseful + FinalrestarantSet$UsrVotesFunny 
#                 + FinalrestarantSet$UsrVotesCool + FinalrestarantSet$NumBizReviews 
#                 + FinalrestarantSet$rev_vote_funny + FinalrestarantSet$rev_vote_useful
#                 + FinalrestarantSet$rev_vote_cool +  FinalrestarantSet$BizCheckIN )

#MSE for full data model
msetrain_regr<-MSE(model_regS, testDataS)
#Show MSE value
msetrain_regr

yhat_reg <- predict(model_regS, testDataS) 
g <- (yhat_reg - testDataS$biz_rating)^2
a <- MSE(model_regS, testDataS)
a


library(ggplot2)
library(reshape2)
x<-seq(0,2927)
df<-data.frame(x,testDataS$biz_rating,yhat_reg)
df2<-melt(data=df,id.vars="x")
ggplot(data=df2, aes(x=x, y=value,colour=variable))+geom_point()

# plot(testDataS$biz_rating)
# lines(yhat_reg,col="red")

#summary(model_reg)
# model coefficients
coefficients(model_regS) 
# CIs for model parameters
confint(model_regS, level=0.95) 
# predicted values
fitted(model_regS) 
# residuals
residuals(model_regS)
# anova table
anova(model_regS)
# covariance matrix for model parameters
vcov(model_regS) 
# regression diagnostics
influence(model_regS) 

# diagnostic plots 
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(model_regS)



#model with some inputs
model_reg1 <- lm(biz_rating~ rev_Star+num_biz_checkin,data=trainDataS)
msetrain_regr1<-MSE(model_reg1, trainDataS)
msetrain_regr1

msetest_regr1<-MSE(model_reg1, testDataS)
msetest_regr1
plot(model_reg1)

yhat_reg1 <- predict(model_reg1, testDataS) 
x<-seq(0,2927)
df<-data.frame(x,testDataS$biz_rating,yhat_reg1)
df2<-melt(data=df,id.vars="x")
ggplot(data=df2, aes(x=x, y=value,colour=variable))+geom_point()

library(leaps)
bestss <- regsubsets(biz_rating~.,data=testDataS,nbest=10)
summary(bestss)
