#include <vector>
#include <limits.h> // for INT_MIN
using namespace std;
// Problem: Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        long long maxi = INT_MIN;
        long long sum = 0;

        for(int i=0;i<nums.size();i++){
            sum += nums[i];
            if(sum > maxi){
                maxi = sum;
            }
            if(sum < 0){
                sum =0;
            }
        }
        max(maxi , sum);  // instead of this .since maxi is int_min . if arr is empty we will return 0 instead of int_min.
        return maxi;
    }
};