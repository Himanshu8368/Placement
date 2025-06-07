#include<vector>
using namespace std;
class Solution{
    public:  
    void solve(int i ,vector<int>&nums,int sum , int k , int& ans){
        if(i>=nums.size()){
            if(sum ==k){
                ans++;
            }
            return;
        }
        sum = sum + nums[i];
        solve(i+1,nums,sum,k,ans);
        sum = sum - nums[i];
        solve(i+1,nums , sum , k , ans);

    }  	
    int countSubsequenceWithTargetSum(vector<int>& nums, int k){
    	int ans =0;
        int sum =0;
        solve(0, nums,0, k  , ans);
        return ans;
    }
};