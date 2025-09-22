#include<stdio.h>
void Print(int *a,int n);
int generate_arr(int* a,int n);
int main(){
    int n;
    scanf("%d",&n);
    int arr[n];
    for(int i=0;i<n;i++){
        arr[i]=i+1;
    }
    do{
        Print(arr,n);
    }while(generate_arr(arr,n));
    return 0;
}
void Print(int *a,int n){
    for(int i=0;i<n;i++){
        printf("%d",a[i]);
    }
    printf("\n");
}
void swap(int *a,int* b){
    int t=*a;
    *a=*b;
    *b=t;
    return;
}
int generate_arr(int* a,int n){
    int r=n-1,l=n-2;
    while(l>=0&&(a[l]>a[l+1])){
        l--;
    }
    if(l<0){
        return 0;
    }
    while(a[l]>a[r]){
        r--;
    }
    swap(&a[r],&a[l++]);
    r=n-1;
    while(r>l){
        swap(&a[r--],&a[l++]);
    }
    return 1;
}
