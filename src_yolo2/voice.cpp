#include <iostream>
#include <sapi.h> //导入语音头文件
#include <string>
//#pragma comment(lib,"sapi.lib") //导入语音头文件库

void  MSSSpeak(LPCTSTR speakContent)// speakContent为LPCTSTR型的字符串,调用此函数即可将文字转为语音
{
	ISpVoice *pVoice = NULL;

	//初始化COM接口

	if (FAILED(::CoInitialize(NULL)))
		MessageBox(NULL, (LPCWSTR)L"COM接口初始化失败！", (LPCWSTR)L"提示", MB_ICONWARNING | MB_CANCELTRYCONTINUE | MB_DEFBUTTON2);

	//获取SpVoice接口

	HRESULT hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);


	if (SUCCEEDED(hr))
	{
		pVoice->SetVolume((USHORT)100); //设置音量，范围是 0 -100
		pVoice->SetRate(0); //设置速度，范围是 -10 - 10
		hr = pVoice->Speak(speakContent, 0, NULL);

		pVoice->Release();

		pVoice = NULL;
	}

	//释放com资源
	::CoUninitialize();
}