# header----
save(list=ls(all=T),file='HW3.RData')
setwd("~/Google Drive/Purdue University/Academics/Sem-3/STAT545/HW3")
load("~/Google Drive/Purdue University/Academics/Sem-3/STAT545/HW3/HW3.RData")

# mnist loaders----
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('mnist/train-images-idx3-ubyte')
  test <<- load_image_file('mnist/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('mnist/train-labels-idx1-ubyte')
  test$y <<- load_label_file('mnist/t10k-labels-idx1-ubyte')  
}
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# data----
load_mnist()
digits<-head(train[[2]], 1000)
labels<-head(train[[3]], 1000)
rm(train, test)

# functions----

my_kmeans<-function(data, k, N, seed=9)
{
  set.seed(seed)
  store.param<-replicate(N, list)
  for(outermost in 1:N)
  {
    centroid<-replicate(k, list)
    k_loss<-list()
    store<-cbind(sample.int(k, dim(data)[1], replace=TRUE), data)
    total<-1
    ctr<-0
    while(total!=0)
    {
      for(i in 1:k)
      {
        subset<-store[store[,1]==i,seq(2,dim(store)[2])]
        if(!is.null(dim(subset)[1]) && dim(subset)[1]!=0){
          centroid[[i]]<-colSums(subset)/dim(subset)[1]
        }else{
          centroid[[i]]<-vector(mode="numeric", length=length(seq(2,dim(store)[2]))) 
        }
        rm(subset)
      }
      previous<-store
      for(i in 1:dim(store)[1])
      {
        edist<-vector(mode="numeric", length=k)
        for(j in 1:length(centroid))
        {
          sq<-(store[i,seq(2,dim(store)[2])]-centroid[[j]])**2
          edist[j]<-sum(sq)
          rm(sq)
        }
        store[i,1]<-which.min(edist)
        rm(edist)
      }
      total<-sum(store[,1]-previous[,1])
      ctr<-ctr+1
      k_loss<-c(k_loss, loss(store))
    }
    store.param[[outermost]]<-list(cluster.assignment=store, loss=loss(store), centroid=centroid, k_loss=unlist(k_loss), iter=ctr)
  }
  minimum_loss<-which.min(unlist(lapply(store.param, function(x) x[[2]])))
  return(list(best=store.param[[minimum_loss]],data=store.param))
}

my_kmedoids<-function(data, k, N, seed=9)
{
  set.seed(seed)
  store.param<-replicate(N, list)
  for(outermost in 1:N)
  {
    medoids<-replicate(k, list)
    k_loss<-list()
    store<-cbind(sample.int(k, dim(data)[1], replace=TRUE), data)
    total<-1 #  stopping criteria
    ctr<-0
    while(total!=0)
    {
      for(i in 1:k)
      {
        subset<-store[store[,1]==i,seq(2,dim(store)[2])]
        if(!is.null(dim(subset)[1]) && dim(subset)[1]!=0){
          # dist_matrix<-as.matrix(dist(subset))
          # dist_matrix_colsums<-colSums(dist_matrix)
          # k_min<-which.min(dist_matrix_colsums)
          # medoids[[i]]<-subset[k_min,]
          medoids[[i]]<-get_prototype(subset)
        }else{
          medoids[[i]]<-vector(mode="numeric", length=length(seq(2,dim(store)[2]))) 
        }
        rm(subset)
      }
      previous<-store
      for(i in 1:dim(store)[1])
      {
        edist<-vector(mode="numeric", length=k)
        for(j in 1:length(medoids))
        {
          sq<-(store[i,seq(2,dim(store)[2])]-medoids[[j]])**2
          edist[j]<-sum(sq)
          rm(sq)
        }
        store[i,1]<-which.min(edist)
        rm(edist)
      }
      total<-sum(store[,1]-previous[,1])
      ctr<-ctr+1
      k_loss<-c(k_loss, loss(store))
    }
    store.param[[outermost]]<-list(cluster.assignment=store, loss=loss(store), medoids=medoids, k_loss=unlist(k_loss), iter=ctr)
  }
  minimum_loss<-which.min(unlist(lapply(store.param, function(x) x[[2]])))
  return(list(best=store.param[[minimum_loss]],data=store.param))
}

# helper functions----
col_mean<-function(mat)
{
  obs<-dim(mat)[1]
  mat<-colSums(mat)
  value<-mat/obs
  return(value)
}
loss<-function(mat)
{
  k<-sort(unique(mat[,1]))
  sum_total<-0
  for(i in 1:length(k))
  {
    sub.set<-mat[mat[,1]==i,seq(2,dim(mat)[2])]
    edist<-as.matrix(dist(sub.set))
    lower.triangle<-lower.tri(edist)
    sum_total<-sum_total + (sum(edist[lower.triangle])/dim(sub.set)[1])
  }
  return(sum_total)
}
plot_clustered_digits<-function(obj,row=NULL,column=NULL)
{
  obj<-obj[["best"]][[3]]
  if(!is.null(row) & !is.null(column))
  {
    par(mfrow=c(row,column))
    K<-sort(length(obj))
    for(i in 1:K)
    {
      show_digit(obj[[i]])
    }
  }else{
    K<-sort(length(obj))
    for(i in 1:K)
    {
      show_digit(obj[[i]])
    }
  }
}
get_prototype<-function(subset)
{
  dist_matrix<-as.matrix(dist(subset))      # Medoid calculation
  dist_matrix_colsums<-colSums(dist_matrix) # Medoid calculation
  k_min<-which.min(dist_matrix_colsums)     # Medoid calculation
  return(subset[k_min,])                    # Medoid calculation
}


# operations----
clustered.digits.k5<-my_kmeans(digits,5, 25)
clustered.digits.k10<-my_kmeans(digits,10, 25)
clustered.digits.k20<-my_kmeans(digits,20, 25)


clustered.digits.k5.medoids<-my_kmedoids(digits, 5, 25)
clustered.digits.k10.medoids<-my_kmedoids(digits, 10, 25)
clustered.digits.k20.medoids<-my_kmedoids(digits, 20, 25)

# plots----

# mean

png("plots/K20.png", width = 10, height = 10, units = 'in', res = 300)
plot_clustered_digits(clustered.digits.k20,5,4)
dev.off()

png("plots/K10.png", width = 10, height = 10, units = 'in', res = 300)
plot_clustered_digits(clustered.digits.k10,3,4)
dev.off()

png("plots/K5.png", width = 10, height = 10, units = 'in', res = 300)
plot_clustered_digits(clustered.digits.k5,3,2)
dev.off()

# medoids

png("plots/K20_medoids.png", width = 10, height = 10, units = 'in', res = 300)
plot_clustered_digits(clustered.digits.k20.medoids,5,4)
dev.off()

png("plots/K10_medoids.png", width = 10, height = 10, units = 'in', res = 300)
plot_clustered_digits(clustered.digits.k10.medoids,3,4)
dev.off()

png("plots/K5_medoids.png", width = 10, height = 10, units = 'in', res = 300)
plot_clustered_digits(clustered.digits.k5.medoids,3,2)
dev.off()
