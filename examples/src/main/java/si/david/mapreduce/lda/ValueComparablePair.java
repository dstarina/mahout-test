/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package si.david.mapreduce.lda;

import java.io.Serializable;

/** A simple (ordered) pair of two objects. Elements may be null.
 * Copy of org.apache.mahout.common.Pair, except for ordering, which is done by values */
public final class ValueComparablePair<A,B> implements Comparable<ValueComparablePair<A,B>>, Serializable {

  private final A first;
  private final B second;

  public ValueComparablePair(A first, B second) {
    this.first = first;
    this.second = second;
  }
  
  public A getFirst() {
    return first;
  }
  
  public B getSecond() {
    return second;
  }
  
  public ValueComparablePair<B, A> swap() {
    return new ValueComparablePair<>(second, first);
  }

  public static <A,B> ValueComparablePair<A,B> of(A a, B b) {
    return new ValueComparablePair<>(a, b);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof ValueComparablePair<?, ?>)) {
      return false;
    }
    ValueComparablePair<?, ?> otherPair = (ValueComparablePair<?, ?>) obj;
    return isEqualOrNulls(first, otherPair.getFirst())
        && isEqualOrNulls(second, otherPair.getSecond());
  }
  
  private static boolean isEqualOrNulls(Object obj1, Object obj2) {
    return obj1 == null ? obj2 == null : obj1.equals(obj2);
  }
  
  @Override
  public int hashCode() {
    int firstHash = hashCodeNull(first);
    // Flip top and bottom 16 bits; this makes the hash function probably different
    // for (a,b) versus (b,a)
    return (firstHash >>> 16 | firstHash << 16) ^ hashCodeNull(second);
  }
  
  private static int hashCodeNull(Object obj) {
    return obj == null ? 0 : obj.hashCode();
  }
  
  @Override
  public String toString() {
    return '(' + String.valueOf(first) + ',' + second + ')';
  }

  /**
   * Defines an ordering on pairs that sorts by second value's natural ordering.
   *
   * @throws ClassCastException if types are not actually {@link Comparable}
   */
  @Override
  public int compareTo(ValueComparablePair<A,B> other) {
    Comparable<B> thisSecond = (Comparable<B>) second;
    B thatSecond = other.getSecond();
    return thisSecond.compareTo(thatSecond);
  }

}
