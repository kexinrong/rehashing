#ifndef __UTIL_H__
#define __UTIL_H__

template<class A, class B, class C>
class triple {
public:
  A first;
  B second;
  C third;

  bool operator<(const triple &a) const {
    if(first < a.first){
      return true;
    } else if (first == a.first) {
      if( second < a.second ) {
        return true;
      } else if( second == a.second ) {
        if(third < a.third)
          return true;
        else
          return false;
      } else {
        return false;
      }
    }
    else return false;
  }
/*
  void operator=(const triple<A,B,C>& a) {
    first = a.first;
    second = a.second;
    third = a.third;
  }
*/

  static bool firstLess( triple<A,B,C> a, triple<A,B,C> b ){
    return a.first < b.first;
  }

};





#endif


