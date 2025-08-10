#include <vector>
#include <algorithm>
using namespace std;
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int ans =INT_MIN;
        int n = nums.size();
        for(int i=0;i<n;i++){
            int multiple = nums[i];
            for(int j=i;j<n;j++){
                if(j>i){
                    multiple= multiple * nums[j];
                    
                }
                ans = max(ans , multiple);
            }
        }
        return ans ;
    }
};

class Solution1 {
public:
    int maxProduct1(vector<int>& nums) {
        int min_p = nums[0];
        int max_p = nums[0];
        int ans = nums[0];
        for(int i=1;i<nums.size();i++){
            if(nums[i]<0){
                swap(min_p,max_p);
            }
            min_p = min(nums[i],min_p*nums[i]);
            max_p = max(nums[i],max_p *nums[i]);
            ans = max(ans,max_p);
        }
        return ans ;
    }



};


