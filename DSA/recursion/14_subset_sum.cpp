#include <vector>
using namespace std;
class Solution {
  public:
    void solve(int i,int sum,vector<int> & nums,vector<int> & ans){
      if(i>=nums.size()){
        ans.push_back(sum);
        return;
      }
      sum+= nums[i];
      solve(i+1,sum,nums,ans);
      sum -= nums[i];
      solve(i+1,sum,nums,ans);
    }
    vector<int> subsetSums(vector<int>& nums) {
        vector<int> ans;
        solve(0,0,nums,ans);
        return ans;
    }
};