#Program to apply neural network to Yelp dataset
#Previuosly You should be run YelpRegression 

#Feedforward neural networks
seed.val<-2
library(nnet)
#set.seed(seed.val)
mod_nnet1<-nnet(biz_rating~.,data=trainDataS,size=10,linout=T)
yhat_nnet <- predict(mod_nnet1,testDataS)
mse_nnet1 <- MSE_WM(yhat_nnet,testDataS$biz_rating)
mse_nnet1
library(devtools)
source_url('https://gist.github.com/fawda123/7471137/raw/c720af2cea5f312717f020a09946800d55b8f45b/nnet_plot_update.r')
plot.nnet(mod_nnet1)

df<-data.frame(x,testDataS$biz_rating,yhat_nnet)
df2<-melt(data=df,id.vars="x")
ggplot(data=df2, aes(x=x, y=value,colour=variable))+geom_point()


#neuralnet function from neuralnet package, notice use of only one response
library(neuralnet)
form.in<-as.formula('rev_stars~rev_length+star_diff+useful_vote+funny_vote+cool_vote+stars_buz+reviewCount_buz+AvgStarUsr')
#set.seed(seed.val)
#mod_nnet2<-neuralnet(form.in,data=mydata,hidden=10)
mod_nnet2<-neuralnet(form.in,mydata,hidden=10, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)
borrar<-x[,-3]
yhat_nnet2 <- compute(mod_nnet2,borrar )
mse_nnet2 <- MSE_WM(rev_stars, yhat_nnet2)
mse_nnet2

#mlp function from RSNNS package
library(RSNNS)
library(Rcpp)
#set.seed(seed.val)
mod_nnet3<-mlp(rev_length+star_diff+freshness+useful_vote+funny_vote+cool_vote+stars_buz+reviewCount_buz+AvgStarUsr, rev_stars, size=10,linOut=T)
