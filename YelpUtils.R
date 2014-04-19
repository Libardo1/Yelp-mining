removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)

removeWords.PlainTextDocument <- function (x, words)
  gsub(sprintf("(*UCP)\\b(%s)\\b", paste(words, collapse = "|")), "", x, 
       perl=TRUE)
