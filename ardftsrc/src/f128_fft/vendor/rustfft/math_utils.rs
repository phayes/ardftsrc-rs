use num_traits::{One, PrimInt, Zero};

pub fn primitive_root(prime: u64) -> Option<u64> {
    let test_exponents: Vec<u64> = distinct_prime_factors(prime - 1)
        .iter()
        .map(|factor| (prime - 1) / factor)
        .collect();
    'next: for potential_root in 2..prime {
        // for each distinct factor, if potential_root^(p-1)/factor mod p is 1, reject it
        for exp in &test_exponents {
            if modular_exponent(potential_root, *exp, prime) == 1 {
                continue 'next;
            }
        }

        // if we reach this point, it means this root was not rejected, so return it
        return Some(potential_root);
    }
    None
}

/// computes base^exponent % modulo using the standard exponentiation by squaring algorithm
pub fn modular_exponent<T: PrimInt>(mut base: T, mut exponent: T, modulo: T) -> T {
    let one = T::one();

    let mut result = one;

    while exponent > Zero::zero() {
        if exponent & one == one {
            result = result * base % modulo;
        }
        exponent = exponent >> One::one();
        base = (base * base) % modulo;
    }

    result
}

/// return all of the prime factors of n, but omit duplicate prime factors
pub fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut result = Vec::new();

    // handle 2 separately so we dont have to worry about adding 2 vs 1
    if n % 2 == 0 {
        while n % 2 == 0 {
            n /= 2;
        }
        result.push(2);
    }
    if n > 1 {
        let mut divisor = 3;
        let mut limit = (n as f32).sqrt() as u64 + 1;
        while divisor < limit {
            if n % divisor == 0 {
                // remove as many factors as possible from n
                while n % divisor == 0 {
                    n /= divisor;
                }
                result.push(divisor);

                // recalculate the limit to reduce the amount of work we need to do
                limit = (n as f32).sqrt() as u64 + 1;
            }

            divisor += 2;
        }

        if n > 1 {
            result.push(n);
        }
    }

    result
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct PrimeFactor {
    pub value: usize,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct PrimeFactors {
    other_factors: Vec<PrimeFactor>,
    n: usize,
    power_two: u32,
    power_three: u32,
    total_factor_count: u32,
    distinct_factor_count: u32,
}
impl PrimeFactors {
    pub fn compute(mut n: usize) -> Self {
        let mut result = Self {
            other_factors: Vec::new(),
            n,
            power_two: 0,
            power_three: 0,
            total_factor_count: 0,
            distinct_factor_count: 0,
        };

        // compute powers of two separately
        result.power_two = n.trailing_zeros();
        result.total_factor_count += result.power_two;
        n >>= result.power_two;
        if result.power_two > 0 {
            result.distinct_factor_count += 1;
        }

        // also compute powers of three separately
        while n % 3 == 0 {
            result.power_three += 1;
            n /= 3;
        }
        result.total_factor_count += result.power_three;
        if result.power_three > 0 {
            result.distinct_factor_count += 1;
        }

        // if we have any other factors, gather them in the "other factors" vec
        if n > 1 {
            let mut divisor = 5;
            // compute divisor limit. if our divisor goes above this limit, we know we won't find any more factors. we'll revise it downwards as we discover factors.
            let mut limit = (n as f32).sqrt() as usize + 1;
            while divisor < limit {
                // Count how many times this divisor divesthe remaining input
                let mut count = 0;
                while n % divisor == 0 {
                    n /= divisor;
                    count += 1;
                }

                // If this entry is actually a divisor of the given number, add it to the array
                if count > 0 {
                    result.other_factors.push(PrimeFactor {
                        value: divisor,
                        count,
                    });
                    result.total_factor_count += count;
                    result.distinct_factor_count += 1;

                    // recalculate the limit to reduce the amount of other factors we need to check
                    limit = (n as f32).sqrt() as usize + 1;
                }

                divisor += 2;
            }

            // because of our limit logic, there might be one factor left
            if n > 1 {
                result
                    .other_factors
                    .push(PrimeFactor { value: n, count: 1 });
                result.total_factor_count += 1;
                result.distinct_factor_count += 1;
            }
        }

        result
    }

    pub fn is_prime(&self) -> bool {
        self.total_factor_count == 1
    }
    pub fn get_product(&self) -> usize {
        self.n
    }
    #[allow(unused)]
    pub fn get_total_factor_count(&self) -> u32 {
        self.total_factor_count
    }
    #[allow(unused)]
    pub fn get_distinct_factor_count(&self) -> u32 {
        self.distinct_factor_count
    }
    #[allow(unused)]
    pub fn get_power_of_two(&self) -> u32 {
        self.power_two
    }
    #[allow(unused)]
    pub fn get_power_of_three(&self) -> u32 {
        self.power_three
    }
    #[allow(unused)]
    pub fn get_other_factors(&self) -> &[PrimeFactor] {
        &self.other_factors
    }
    #[allow(unused)]
    pub fn is_power_of_three(&self) -> bool {
        self.power_three > 0 && self.power_two == 0 && self.other_factors.len() == 0
    }

    // Divides the number by the given prime factor. Returns None if the resulting number is one.
    #[allow(unused)]
    pub fn remove_factors(mut self, factor: PrimeFactor) -> Option<Self> {
        if factor.count == 0 {
            return Some(self);
        }
        if factor.value == 2 {
            self.power_two = self.power_two.checked_sub(factor.count).unwrap();
            self.n >>= factor.count;
            self.total_factor_count -= factor.count;
            if self.power_two == 0 {
                self.distinct_factor_count -= 1;
            }
            if self.n > 1 {
                return Some(self);
            }
        } else if factor.value == 3 {
            self.power_three = self.power_three.checked_sub(factor.count).unwrap();
            self.n /= 3.pow(factor.count);
            self.total_factor_count -= factor.count;
            if self.power_two == 0 {
                self.distinct_factor_count -= 1;
            }
            if self.n > 1 {
                return Some(self);
            }
        } else {
            let found_factor = self
                .other_factors
                .iter_mut()
                .find(|item| item.value == factor.value)
                .unwrap();
            found_factor.count = found_factor.count.checked_sub(factor.count).unwrap();
            self.n /= factor.value.pow(factor.count);
            self.total_factor_count -= factor.count;
            if found_factor.count == 0 {
                self.distinct_factor_count -= 1;
                self.other_factors.retain(|item| item.value != factor.value);
            }
            if self.n > 1 {
                return Some(self);
            }
        }
        None
    }

    // returns true if we have any factors whose value is less than or equal to the provided factor
    pub fn has_factors_leq(&self, factor: usize) -> bool {
        self.power_two > 0
            || self.power_three > 0
            || self
                .other_factors
                .first()
                .map_or(false, |f| f.value <= factor)
    }

    // returns true if we have any factors whose value is greater than the provided factor
    pub fn has_factors_gt(&self, factor: usize) -> bool {
        (factor < 2 && self.power_two > 0)
            || (factor < 3 && self.power_three > 0)
            || self
                .other_factors
                .last()
                .map_or(false, |f| f.value > factor)
    }

    // returns the product of all factors greater than the provided min_factor
    pub fn product_above(&self, min_factor: usize) -> usize {
        self.other_factors
            .iter()
            .skip_while(|f| f.value <= min_factor)
            .map(|f| f.value.pow(f.count))
            .product()
    }

    // Splits this set of prime factors into two different sets so that the products of the two sets are as close as possible
    pub fn partition_factors(mut self) -> (Self, Self) {
        // Make sure this isn't a prime number
        assert!(!self.is_prime());

        // If the given length is a perfect square, put the square root into both returned arays
        if self.power_two % 2 == 0
            && self.power_three % 2 == 0
            && self
                .other_factors
                .iter()
                .all(|factor| factor.count % 2 == 0)
        {
            let mut new_product = 1;

            // cut our power of two in half
            self.power_two /= 2;
            new_product <<= self.power_two;

            // cout our power of three in half
            self.power_three /= 2;
            new_product *= 3.pow(self.power_three);

            // cut all our other factors in half
            for factor in self.other_factors.iter_mut() {
                factor.count /= 2;
                new_product *= factor.value.pow(factor.count);
            }

            // update our cached properties and return 2 copies of ourself
            self.total_factor_count /= 2;
            self.n = new_product;
            (self.clone(), self)
        } else if self.distinct_factor_count == 1 {
            // If there's only one factor, just split it as evenly as possible
            let mut half = Self {
                other_factors: Vec::new(),
                n: self.n,
                power_two: self.power_two / 2,
                power_three: self.power_three / 2,
                total_factor_count: self.total_factor_count / 2,
                distinct_factor_count: 1,
            };

            // We computed one half via integer division -- compute the other half by subtracting the divided values fro mthe original
            self.power_two -= half.power_two;
            self.power_three -= half.power_three;
            self.total_factor_count -= half.total_factor_count;

            // Update the product values for each half, with different logic depending on what kind of single factor we have
            if let Some(first_factor) = self.other_factors.first_mut() {
                // we actualyl skipped updating the "other factor"  earlier, so cut it in half and do the subtraction now
                assert!(first_factor.count > 1); // If this is only one, then we're prime. we passed the "is_prime" assert earlier, so that would be a contradiction
                let half_factor = PrimeFactor {
                    value: first_factor.value,
                    count: first_factor.count / 2,
                };
                first_factor.count -= half_factor.count;
                half.other_factors.push(half_factor);

                self.n = first_factor.value.pow(first_factor.count);
                half.n = half_factor.value.pow(half_factor.count);
            } else if half.power_two > 0 {
                half.n = 1 << half.power_two;
                self.n = 1 << self.power_two;
            } else if half.power_three > 0 {
                half.n = 3.pow(half.power_three);
                self.n = 3.pow(self.power_three);
            }

            (self, half)
        } else {
            // we have a mixed bag of products. we're going to greedily try to evenly distribute entire groups of factors in one direction or the other
            let mut left_product = 1;
            let mut right_product = 1;

            // for each factor, put it in whichever cumulative half is smaller
            for factor in self.other_factors {
                let factor_product = factor.value.pow(factor.count as u32);
                if left_product <= right_product {
                    left_product *= factor_product;
                } else {
                    right_product *= factor_product;
                }
            }
            if left_product <= right_product {
                left_product <<= self.power_two;
            } else {
                right_product <<= self.power_two;
            }
            if self.power_three > 0 && left_product <= right_product {
                left_product *= 3.pow(self.power_three);
            } else {
                right_product *= 3.pow(self.power_three);
            }

            // now that we have our two products, compute a prime factorization for them
            // we could maintain factor lists internally to save some computation and an allocation, but it led to a lot of code and this is so much simpler
            (Self::compute(left_product), Self::compute(right_product))
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PartialFactors {
    power2: u32,
    power3: u32,
    power5: u32,
    power7: u32,
    power11: u32,
    other_factors: usize,
}
impl PartialFactors {
    #[allow(unused)]
    pub fn compute(len: usize) -> Self {
        let power2 = len.trailing_zeros();
        let mut other_factors = len >> power2;

        let mut power3 = 0;
        while other_factors % 3 == 0 {
            power3 += 1;
            other_factors /= 3;
        }

        let mut power5 = 0;
        while other_factors % 5 == 0 {
            power5 += 1;
            other_factors /= 5;
        }

        let mut power7 = 0;
        while other_factors % 7 == 0 {
            power7 += 1;
            other_factors /= 7;
        }

        let mut power11 = 0;
        while other_factors % 11 == 0 {
            power11 += 1;
            other_factors /= 11;
        }

        Self {
            power2,
            power3,
            power5,
            power7,
            power11,
            other_factors,
        }
    }

    #[allow(unused)]
    pub fn get_power2(&self) -> u32 {
        self.power2
    }
    #[allow(unused)]
    pub fn get_power3(&self) -> u32 {
        self.power3
    }
    #[allow(unused)]
    pub fn get_power5(&self) -> u32 {
        self.power5
    }
    #[allow(unused)]
    pub fn get_power7(&self) -> u32 {
        self.power7
    }
    #[allow(unused)]
    pub fn get_power11(&self) -> u32 {
        self.power11
    }
    #[allow(unused)]
    pub fn get_other_factors(&self) -> usize {
        self.other_factors
    }
    #[allow(unused)]
    pub fn product(&self) -> usize {
        (self.other_factors
            * 3.pow(self.power3)
            * 5.pow(self.power5)
            * 7.pow(self.power7)
            * 11.pow(self.power11))
            << self.power2
    }
    #[allow(unused)]
    pub fn product_power2power3(&self) -> usize {
        3.pow(self.power3) << self.power2
    }
    #[allow(unused)]
    pub fn divide_by(&self, divisor: &PartialFactors) -> Option<PartialFactors> {
        let two_divides = self.power2 >= divisor.power2;
        let three_divides = self.power3 >= divisor.power3;
        let five_divides = self.power5 >= divisor.power5;
        let seven_divides = self.power7 >= divisor.power7;
        let eleven_divides = self.power11 >= divisor.power11;
        let other_divides = self.other_factors % divisor.other_factors == 0;
        if two_divides
            && three_divides
            && five_divides
            && seven_divides
            && eleven_divides
            && other_divides
        {
            Some(Self {
                power2: self.power2 - divisor.power2,
                power3: self.power3 - divisor.power3,
                power5: self.power5 - divisor.power5,
                power7: self.power7 - divisor.power7,
                power11: self.power11 - divisor.power11,
                other_factors: if self.other_factors == divisor.other_factors {
                    1
                } else {
                    self.other_factors / divisor.other_factors
                },
            })
        } else {
            None
        }
    }
}

