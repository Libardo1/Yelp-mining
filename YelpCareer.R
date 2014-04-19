library("rjson", lib.loc="/Library/Frameworks/R.framework/Versions/3.0/Resources/library")
library("jsonlite")
library(AppliedPredictiveModeling)
library(caret)

setwd('/Users/prateek/DataSets/YELP_Career/yelp_training_set/truncatedSet')

business_json_file <- "yelp_training_set_business.json"
review_json_file   <- "yelp_training_set_review.json"
user_json_file     <- "yelp_training_set_user.json"
checkin_json_file  <- "yelp_training_set_checkin.json"

#########################LOADING DATA##############################################################
bus_data <- jsonlite::fromJSON(sprintf("[%s]", paste(readLines(business_json_file),collapse=",")))
rev_data <- jsonlite::fromJSON(sprintf("[%s]", paste(readLines(review_json_file),collapse=",")))
usr_data <- jsonlite::fromJSON(sprintf("[%s]", paste(readLines(user_json_file),collapse=",")))
chk_data <- jsonlite::fromJSON(sprintf("[%s]", paste(readLines(checkin_json_file),collapse=",")))


#########################FEATURE SET DESIGN########################################################
###################################################################################################
# Freq calculation, Topic detection ; Mostly useless right now, but has potential to be used w/ other stuff.
###################################################################################################
#Convert reviews to dataframes
library(tm)
library(SnowballC)
library(sqldf)
source("/Users/prateek/DataSets/YELP_Career/YelpUtils.R")

usr_sqldf <- data.frame(user_id=usr_data$user_id, review_count=usr_data$review_count, 
                        average_stars=usr_data$average_stars, vote_funny=usr_data$votes$funny, 
                        vote_useful=usr_data$votes$useful, vote_cool=usr_data$votes$cool)
rev_sqldf <- data.frame(user_id=rev_data$user_id, review_id=rev_data$review_id, 
                        stars=rev_data$stars, business_id=rev_data$business_id,
                        text=rev_data$text, date=rev_data$date, 
                        vote_funny=rev_data$votes$funny, vote_useful=rev_data$votes$useful, 
                        vote_cool=rev_data$votes$cool)

bus_sqldf <- data.frame(business_id=bus_data$business_id, longitude=bus_data$longitude,
                        latitude=bus_data$latitude, stars=bus_data$stars,
                        review_count=bus_data$review_count) # leaving out categories & open for now. Might have to add it back later. 

sum <- rep(0, dim(chk_data)[1])
# This is not correct, need to fix it later...
for(i in 1:dim(chk_data)[1]){
  for(j in 1:length(chk_data[i,]$checkin_info)){
    if(!is.na(chk_data[i,]$checkin_info[1,j])){
      sum[i] = sum[i] + chk_data$checkin_info[1,j]
    }
    sum[is.na(sum)] <- 0
    
  } 
}

chk_sqldf <- data.frame(chk_data$business_id, sum)

# This is done to preserve the order of my dataset when being used for MLR or other studff...
rev_sqldf <- rev_sqldf[order(rev_sqldf$review_id),]

myCorpus <- Corpus(VectorSource(rev_sqldf$text,  encoding='UTF-8'))
# following line is needed to remove stop words, else it was failing on MAC. Might have to adjust for other machines.
myCorpus <- tm_map(myCorpus, function(x) iconv(x, to='UTF-8-MAC', sub='byte'))
myCorpus <- tm_map(myCorpus, removePunctuation)
myCorpus <- tm_map(myCorpus, tolower, mc.cores=1, mc.preschedule=TRUE)
myCorpus <- tm_map(myCorpus, removeNumbers)
myCorpus <- tm_map(myCorpus, removeURL, mc.cores=1, mc.preschedule=TRUE)
# remove 'r' and 'big' from stopwords
myStopwords <- setdiff(stopwords("english"), c("r", "big"))
# remove stopwords
myCorpus <- tm_map(myCorpus, removeWords.PlainTextDocument, myStopwords,  mc.cores=1, mc.preschedule=TRUE)
# keep a copy of corpus
myCorpusCopy <- myCorpus
# stem words
myCorpus <- tm_map(myCorpus, stemDocument)
# stem completion
myCorpus <- tm_map(myCorpus, stemCompletion, dictionary = myCorpusCopy)
# Frequent terms
myTdm <- TermDocumentMatrix(myCorpus, control=list(wordLengths=c(1,Inf)))
inspect(myTdm)
freq.terms <- findFreqTerms(myTdm, lowfreq=20)
# What are my top 1000 words that i need to look at?
m <- as.matrix(myTdm)
freq <- sort(rowSums(m), decreasing=TRUE)
freq[1:5]
str(attributes(freq))

# Drawing word cloud
library(wordcloud)
wordcloud(words=names(freq), freq=freq, min.freq=20, random.order=F)

# Cool Metrics can be obtained...
findAssocs(myTdm, "food", 0.25)

# Topic Modeling
library(topicmodels)
myLda <- LDA(as.DocumentTermMatrix(myTdm), k=8)
terms(myLda, 5)
# plot(myTdm, term=freq.terms, corThreshold=0.1, weighting=T) ######Very Intensive
edit(myCorpus)
edit(freq.terms)

#########################FEATURE SET DESIGN########################################################
###################################################################################################
# Length of Review by using "myCorpusCopy" from above.
###################################################################################################
rev_length <- colSums(m) # sum of each review stored in a vector

#########################FEATURE SET DESIGN########################################################
###################################################################################################
# Stars of each Review by user
###################################################################################################
rev_stars <- rev_sqldf$stars # star rating per review

#########################FEATURE SET DESIGN########################################################
###################################################################################################
# review_count of the user who reviewed this review
###################################################################################################
rev_usr_join <- sqldf("SELECT usr_sqldf.review_count, rev_sqldf.review_id, rev_sqldf.user_id FROM rev_sqldf 
                      LEFT JOIN  usr_sqldf on rev_sqldf.user_id=usr_sqldf.user_id")
rev_usr_join$review_count[is.na(rev_usr_join$review_count) ] <- 0 

#########################FEATURE SET DESIGN########################################################
###################################################################################################
# Bias: Avg. Star of business - Avg. Star given by the user
###################################################################################################
rev_biz_join <- sqldf("SELECT bus_sqldf.stars as biz_stars, bus_sqldf.review_count, rev_sqldf.business_id, rev_sqldf.review_id, rev_sqldf.stars as usr_star, rev_sqldf.user_id FROM rev_sqldf 
                      LEFT JOIN  bus_sqldf on rev_sqldf.business_id=bus_sqldf.business_id")
rev_biz_join[is.na(rev_biz_join) ] <- 0 
# Currently, a lot of the reviews are negative because 
star_diff <- rev_biz_join$biz_stars - rev_biz_join$usr_star

# library(corrplot)
# star_diff <- as.matrix(data.frame(rev_biz_join$biz_stars, rev_biz_join$usr_star, rev_biz_join$biz_stars - rev_biz_join$usr_star))
# corrplot(cor(star_diff))

#########################FEATURE SET DESIGN########################################################
###################################################################################################
# Freshness...Date Manupulation...
###################################################################################################
freshness <- abs(max(as.Date(rev_sqldf$date)) - as.Date(rev_sqldf$date) + 1) # to avoid having zeros...



