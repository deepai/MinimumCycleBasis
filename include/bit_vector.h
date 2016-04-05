#ifndef _H_BIT_VECTOR
#define _H_BIT_VECTOR

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
	}

	inline unsigned long long get_xor_number(int &offset,bool &val)
	{
		unsigned long long initial_value = val;
		if(val == false)
			return initial_value;

		initial_value <<= offset;

		return initial_value;
	}

	inline unsigned long long &get_element_for_pos(int &pos)
	{
		int index = pos/64;
		return elements[index];
	}

	inline void set_bit(int pos,bool val)
	{
		unsigned long long &item = get_element_for_pos(pos);
		int offset = pos%64;

		unsigned long long xor_number = get_xor_number(offset,val);

		item = item^xor_number;
	}
};

#endif