#include <iomanip>

#include "Resizer.h"


int Resizer::MaxVectorSizeDecodeInAsync   =  100000;
int Resizer::MaxImageOutSizeResizeInAsync =   10000;
int Resizer::MaxImageOutSizeEncodeInAsync =  100000;

void Resizer::operator()()
{
	unique_lock<mutex> decode_lock(decode_tasks_mutex);
	decode_lock.unlock();
	unique_lock<mutex> resize_lock(resize_tasks_mutex);
	resize_lock.unlock();
	unique_lock<mutex> encode_lock(encode_tasks_mutex);
	encode_lock.unlock();
	unique_lock<mutex> read_lock(read_tasks_mutex);
	read_lock.unlock();
	unique_lock<mutex> write_lock(write_tasks_mutex);
	write_lock.unlock();

	//while (!is_exit() || have_tasks())
	while (!is_exit() || (counter != 0))
	{

		unique_lock<mutex> lock(tasks_mutex);
		if (!have_tasks())
			condition.wait(lock);
		lock.unlock();
		
		cout << "Read   tasks in deq " << setw(2) << read_tasks.size()   << " start " << setw(2) << counter_start_read   << " end " << setw(2) << counter_end_read   << endl;
		cout << "Decode tasks in deq " << setw(2) << decode_tasks.size() << " start " << setw(2) << counter_start_decode << " end " << setw(2) << counter_end_decode << endl;
		cout << "Resize tasks in deq " << setw(2) << resize_tasks.size() << " start " << setw(2) << counter_start_resize << " end " << setw(2) << counter_end_resize << endl;
		cout << "Encode tasks in deq " << setw(2) << encode_tasks.size() << " start " << setw(2) << counter_start_encode << " end " << setw(2) << counter_end_encode << endl;
		cout << "Write  tasks in deq " << setw(2) << write_tasks.size()  << " start " << setw(2) << counter_start_write  << " end " << setw(2) << counter_end_write  << endl << endl;
		
		decode_lock.lock();
		if (!decode_tasks.empty())
		{
			Decode_Task dt = move(decode_tasks.front());
			decode_tasks.pop_front();

			if (dt.vec.size() < MaxVectorSizeDecodeInAsync)
			{				
				future<void> fut2 = std::async(std::launch::async,
					[&]()->void
					{
						packaged_task<cv::Mat(vector<char>)> task = move(dt.pac_t);
						
						counter_start_decode++;
						task(dt.vec);
						counter_end_decode++;
						future<cv::Mat> fut = task.get_future();
						add_resize_task(resize2, move(fut.get()), dt.file_out, dt.size_out);
					}
				);	
			}
			else
			{
				thread thrd([&](Decode_Task dt2)
					{
						packaged_task<cv::Mat(vector<char>)> task = move(dt2.pac_t);
						counter_start_decode++;
						task(dt2.vec);
						counter_end_decode++;
						future<cv::Mat> fut = task.get_future();
						add_resize_task(resize2, fut.get(), dt2.file_out, dt2.size_out);
					},
					move(dt));
				thrd.detach();	
			}
		}
		decode_lock.unlock();

		resize_lock.lock();
		if (!resize_tasks.empty())
		{
			Resize_Task rt = move(resize_tasks.front());
			resize_tasks.pop_front();

			if (rt.size_out.height * rt.size_out.width < MaxImageOutSizeResizeInAsync)
			{				
				future<void> fut2 = async(launch::async,
					[&]()->void
					{
					packaged_task<cv::Mat(cv::Mat, cv::Size)> task = move(rt.pac_t);
					counter_start_resize++;
					task(rt.mat_in, rt.size_out);
					counter_end_resize++;
					future<cv::Mat> fut = task.get_future();
						add_encode_task(mat_encode_to_buffer, move(fut.get()), rt.file_out);
					}
				);				
			}
			else
			{
				thread thrd([&](Resize_Task rt2)
					{
						packaged_task<cv::Mat(cv::Mat, cv::Size)> task = move(rt2.pac_t);
						counter_start_resize++;
						task(rt2.mat_in, rt2.size_out);
						counter_end_resize++;
						future<cv::Mat> fut = task.get_future();
						add_encode_task(mat_encode_to_buffer, move(fut.get()), rt2.file_out);
					},
					move(rt));
				thrd.detach();
			}
		}
		resize_lock.unlock();

		encode_lock.lock();
		if (!encode_tasks.empty())
		{
			Encode_Task et = move(encode_tasks.front());
			encode_tasks.pop_front();

			if (et.mat_out.size().height * et.mat_out.size().width < MaxImageOutSizeEncodeInAsync)
			{				
				future<void> fut2 = async(launch::async,
					[&]()->void
					{
						packaged_task<std::vector<uchar>(cv::Mat)> task = move(et.pac_t);
						counter_start_encode++;
						task(et.mat_out);
						counter_end_encode++;
						future<std::vector<uchar>> fut = task.get_future();
						add_write_task(write_file, move(fut.get()), et.file_out);
					}
				);	
			}
			else
			{
				thread thrd([&](Encode_Task et2)
				{
					packaged_task<std::vector<uchar>(cv::Mat)> task = move(et2.pac_t);
					counter_start_encode++;
					task(et2.mat_out);
					counter_end_encode++;
					future<std::vector<uchar>> fut = task.get_future();
					add_write_task(write_file, move(fut.get()), et2.file_out);
				},
					move(et));
				thrd.detach();
			}
		}
		encode_lock.unlock();

		read_lock.lock();
		if (!read_tasks.empty())
		{
			Read_Task rt = move(read_tasks.front());
			read_tasks.pop_front();

			//future<void> fut2 = async(launch::async,
			//[&]()
			//{
			packaged_task<vector<char>(string)> task = move(rt.pac_t);
			counter_start_read++;
			task(rt.file_in);
			counter_end_read++;
			future<vector<char>> fut = task.get_future();
			add_decode_task(buffer_decode_to_mat, move(fut.get()), rt.file_out, rt.size_out);
			//}
			//);

			counter++;
		}
		read_lock.unlock();

		write_lock.lock();
		if (!write_tasks.empty())
		{
			Write_Task wt = move(write_tasks.front());
			write_tasks.pop_front();

			packaged_task<void(string, vector<uchar>&)> task = move(wt.pac_t);
			counter_start_write++;
			task(wt.file_out, wt.vec);
			counter_end_write++;

			counter--;
		}
		write_lock.unlock();
	}
}

