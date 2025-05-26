#include <vector>
#include <map>
using namespace std;
// This code finds the single number in an array where every other number appears twice.
// The solution uses the XOR operation to find the unique number efficiently.
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int XOR =0;
        for (int i=0;i<nums.size();i++){
            XOR = XOR ^ nums[i];
        }
        return XOR;
    }
};
// Alternative solution using map to count occurrences
class Solution2 {
public:
    int singleNumber(vector<int>& nums) {
        map<int,int> store;
        for(int i =0;i< nums.size();i++){
            store[nums[i]]++;
        }
        for(auto it : store){
            if(it.second ==1){
                return it.first;
            }
        }
        return 0;
        // int XOR =0;
        // for (int i=0;i<nums.size();i++){
        //     XOR = XOR ^ nums[i];
        // }
        // return XOR;
    }
};
// third  we can use array to store the count of each number but it does not work for negative numbers or large ranges
////////////////////////////////////////////////////////////////////////////
// The map-based solution is less efficient than the XOR solution, as it has a higher time complexity due to the use of a map.
// The XOR solution runs in O(n) time and O(1) space, while the map solution runs in O(n) time and O(n) space.
// The XOR solution is preferred for its simplicity and efficiency.
// The XOR operation is used because it has the property that a ^ a = 0 and a ^ 0 = a, which allows us to cancel out pairs of numbers.
// The map solution counts occurrences of each number, which is less efficient but still works.
// The XOR solution is the most efficient way to find the single number in an array where every other number appears twice.