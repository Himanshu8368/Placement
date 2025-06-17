#include <string>
#include <stack>
using namespace std;
// Problem: 151. Reverse Words in a String
// Link: https://leetcode.com/problems/reverse-words-in-a-string/
// Difficulty: Medium

class Solution {
public:
    string reverseWords(string s) {
        string temp ="";
        stack<string> st;
        string ans ;
        for(int i=0;i<s.size();i++){
            if(s[i]==' '){
                if(!temp.empty()){
                    st.push(temp);
                    temp = "";
                    
                }
            }
            else{
                temp +=s[i];
            }
        }
        if(temp != ""){
            st.push(temp);
        }
        while(!st.empty()){
            ans+=st.top();
            st.pop();
            if(!st.empty()){
                ans+=" ";
            }
        }
        return ans;
    }
};