#pragma once

#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <iostream>
#include <deque>
#include <vector>

#include <thread>
#include <future>
#include <atomic>

#include "Funcs.h"

using namespace std;

class Resizer
{
private:
	static int MaxVectorSizeDecodeInAsync;// = 1000000;
	static int MaxImageOutSizeResizeInAsync;
	static int MaxImageOutSizeEncodeInAsync;

	struct Read_Task
	{
		string file_in;
		string file_out;
		cv::Size size_out;
		packaged_task<vector<char>(string)> pac_t;
	};
	deque<Read_Task> read_tasks;
	mutex read_tasks_mutex;

	struct Decode_Task
	{
		vector<char> vec;
		string file_out;
		cv::Size size_out;
		packaged_task<cv::Mat(vector<char>)> pac_t;
	};
	deque<Decode_Task> decode_tasks;
	mutex decode_tasks_mutex;

	struct Resize_Task
	{
		cv::Mat mat_in;
		string file_out;
		cv::Size size_out;
		packaged_task<cv::Mat(cv::Mat, cv::Size)> pac_t;
	};
	deque<Resize_Task> resize_tasks;
	mutex resize_tasks_mutex;

	struct Encode_Task
	{
		cv::Mat mat_out;
		string file_out;
		packaged_task<std::vector<uchar>(cv::Mat)> pac_t;
	};
	deque<Encode_Task> encode_tasks;
	mutex encode_tasks_mutex;

	struct Write_Task
	{
		std::vector<uchar> vec;
		string file_out;
		packaged_task<void(string, vector<uchar>&)> pac_t;
	};
	deque<Write_Task> write_tasks;
	mutex write_tasks_mutex;



	mutex exit_mutex;
	bool exitVal = false;
	bool is_exit()
	{
		lock_guard<mutex> lock(exit_mutex);
		return exitVal;
	}

	bool have_tasks()
	{
		if (!read_tasks.empty() || !decode_tasks.empty() || !resize_tasks.empty() || !encode_tasks.empty() || !write_tasks.empty())
			return true;
		return false;
	}

	atomic<int> counter = 1;

	mutex tasks_mutex;
	condition_variable condition;

	atomic<int> counter_start_read = 0;
	atomic<int> counter_start_decode = 0;
	atomic<int> counter_start_resize = 0;
	atomic<int> counter_start_encode = 0;
	atomic<int> counter_start_write = 0;

	atomic<int> counter_end_read = 0;
	atomic<int> counter_end_decode = 0;
	atomic<int> counter_end_resize = 0;
	atomic<int> counter_end_encode = 0;
	atomic<int> counter_end_write = 0;

	template<typename func>
	void add_read_task(func f, string file_in, string file_out, cv::Size size_out)
	{
		Read_Task rt;
		decltype(rt.pac_t) task(f);
		rt.pac_t = move(task);
		rt.file_in = file_in;
		rt.file_out = file_out;
		rt.size_out = size_out;
		lock_guard<mutex> lock(read_tasks_mutex);
		read_tasks.push_back(move(rt));

		//unique_lock<mutex> lock2(tasks_mutex);
		condition.notify_all();
	}
	
	template<typename func>
	void add_decode_task(func f, vector<char>&& vec, string file_out, cv::Size size_out)
	{
		Decode_Task dt;
		decltype(dt.pac_t) task(f);
		dt.vec = move(vec);
		dt.file_out = file_out;
		dt.size_out = size_out;
		dt.pac_t = move(task);
		lock_guard<mutex> lock(decode_tasks_mutex);
		decode_tasks.push_back(move(dt));

		//unique_lock<mutex> lock2(tasks_mutex);
		condition.notify_all();
	}

	template<typename func>
	void add_resize_task(func f, cv::Mat mat_in, string file_out, cv::Size size_out)
	{
		Resize_Task rt;
		decltype(rt.pac_t) task(f);
		rt.mat_in = mat_in;
		rt.file_out = file_out;
		rt.size_out = size_out;
		rt.pac_t = move(task);
		lock_guard<mutex> lock(resize_tasks_mutex);
		resize_tasks.push_back(move(rt));

		//unique_lock<mutex> lock2(tasks_mutex);
		condition.notify_all();
	}

	template<typename func>
	void add_encode_task(func f, cv::Mat mat_out, string file_out)
	{
		Encode_Task et;
		decltype(et.pac_t) task(f);
		et.file_out = file_out;
		et.mat_out = mat_out;
		et.pac_t = move(task);
		lock_guard<mutex> lock(encode_tasks_mutex);
		encode_tasks.push_back(move(et));

		//unique_lock<mutex> lock2(tasks_mutex);
		condition.notify_all();
	}

	template<typename func>
	void add_write_task(func f, std::vector<uchar> vec, string file_out)
	{
		Write_Task wt;
		decltype(wt.pac_t) task(f);
		wt.pac_t = move(task);
		wt.file_out = file_out;
		wt.vec = move(vec);
		lock_guard<mutex> lock(write_tasks_mutex);
		write_tasks.push_back(move(wt));

		//unique_lock<mutex> lock2(tasks_mutex);
		condition.notify_all();
	}

	

public:
	
	
	Resizer() { }
	~Resizer() { }
	void operator()();

	void add_file_to_resize(string file_in, string file_out, cv::Size size_out)
	{
		add_read_task(read_file, file_in, file_out, size_out);
	}
	
	void setExit()
	{
		lock_guard<mutex> lock(exit_mutex);
		counter--;
		exitVal = true;
	}
	
};
