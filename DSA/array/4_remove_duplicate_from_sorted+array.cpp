#include<iostream>
#include<vector>

using namespace std;

// Function to remove duplicates from a sorted array

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if(nums.size()==0){
            return 0;
        }
        int prev = nums[0];
        int count =1;
        for(int i=1 ; i<nums.size();i++){
            if(nums[i]!=prev){
                count++;
            }
            prev = nums[i];
        }
        return count;
    }
};

    // vector<int> removeDuplicates(vector<int>& nums) {
    //     vector<int> answer;
    //     if(nums.size()==0){
    //         return answer;
    //     }
    //     int prev = nums[0];
    //     answer.push_back(nums[0]);
    //     for(int i=1;i<nums.size();i++){
    //         if(nums[i]!=prev){
    //             answer.push_back(nums[i]);
    //         }
    //         prev = nums[i];
    //     }
    //     return answer;
    // }





