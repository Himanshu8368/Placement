#include<iostream>
#include<vector>    
using namespace std;

class Solution {
public:
    void reverse(vector<int>& nums, int start , int end){
        while(start<=end){
            int temp = nums[start];
            nums[start ]= nums[end];
            nums[end]= temp;
            start++;
            end--;
        }
    }

    void rotate(vector<int>& nums, int k) {
        int array_size = nums.size();
                k =k%array_size;
        if(k>array_size){
            return;
        }
        //reverse last k element
        reverse(nums,array_size -k,array_size -1);
        //reverse start to last k element
        reverse(nums,0,array_size -k-1);
        //reverse the whole array
        reverse(nums,0,array_size -1);
    }
};
