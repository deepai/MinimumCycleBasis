#ifndef _H_BIT_VECTOR
#define _H_BIT_VECTOR

#include <stack>
#include <omp.h>
#include <cstring>

class bit_vector
{

public:

	int num_elements;
	int size;

	unsigned long long *elements;

	bit_vector(int &n)
	{	
		num_elements = n;
		size = (int)(ceil((double)n/64));
		elements = new unsigned long long[size];
		memset(elements,0,sizeof(unsigned long long) * size);
	}

	~bit_vector()
	{
		delete[] elements;
	}

	int get_size()
	{
		return size;
	}

	int get_num_elements()
	{
		return num_elements;
	}

	inline unsigned long long get_or_number(int &offset,bool &val)
	{
		unsigned long long initial_value = val;
		if(val == false)
			return initial_value;

		initial_value <<= offset;

		return initial_value;
	}
	inline unsigned get_and_numbers(unsigned long long &val1, unsigned long long &val2)
	{
		unsigned long long temp = (val1 & val2);
		unsigned result = 0;

		int count = 0;

		while(temp != 0)
		{
			temp -= (temp & -temp);
			count++;
		}

		result = count%2;

		return result;
	}

	//Return the actual index of the element containing the offset.
	inline unsigned long long &get_element_for_pos(int &pos)
	{
		int index = pos/64;
		return elements[index];
	}

	inline print_bits(unsigned long long val)
	{
		std::stack<bool> bits;

		int count = 64;

		while (val || (count >0)) {
		    	if (val & 1)
		        	bits.push(1);
		    	else
		        	bits.push(0);

		    	val >>= 1;
		    	count--;
		}

		while(!bits.empty())
		{
			printf("%d",bits.top());
			bits.pop();
		}
	}

	inline void set_bit(int pos,bool val)
	{
		unsigned long long &item = get_element_for_pos(pos);
		int offset = pos%64;

		unsigned long long or_number = get_or_number(offset,val);

		item = item | or_number;
	}

	void do_xor(bit_vector *vector)
	{
		assert(vector->size == size);

		#pragma omp parallel for
		for(int i=0;i<size;i++)
			elements[i] = elements[i]^vector->elements[i];
	}

	unsigned dot_product(bit_vector *vector1)
	{
		unsigned val = 0;

		for(int i=0;i<size;i++)
		{
			val = (val + get_and_numbers(elements[i],vector1->elements[i])) % 2;
		}

		return val;
	}

	void print()
	{
		for(int i=0;i<size;i++)
		{
			print_bits(elements[i]);
			printf(" ");
		}
		printf("\n");
	}

	//get bit value at the position pos such that pos belongs to [0- num_elements - 1]
	inline unsigned get_bit(int pos)
	{
		unsigned long long &item = get_element_for_pos(pos);
		int offset = pos%64;

		unsigned val = (item >> offset) & 1;
		return val;
	}
};

#endif