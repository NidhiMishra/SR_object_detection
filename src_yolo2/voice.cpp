#include <iostream>
#include <sapi.h> //��������ͷ�ļ�
#include <string>
//#pragma comment(lib,"sapi.lib") //��������ͷ�ļ���

void  MSSSpeak(LPCTSTR speakContent)// speakContentΪLPCTSTR�͵��ַ���,���ô˺������ɽ�����תΪ����
{
	ISpVoice *pVoice = NULL;

	//��ʼ��COM�ӿ�

	if (FAILED(::CoInitialize(NULL)))
		MessageBox(NULL, (LPCWSTR)L"COM�ӿڳ�ʼ��ʧ�ܣ�", (LPCWSTR)L"��ʾ", MB_ICONWARNING | MB_CANCELTRYCONTINUE | MB_DEFBUTTON2);

	//��ȡSpVoice�ӿ�

	HRESULT hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);


	if (SUCCEEDED(hr))
	{
		pVoice->SetVolume((USHORT)100); //������������Χ�� 0 -100
		pVoice->SetRate(0); //�����ٶȣ���Χ�� -10 - 10
		hr = pVoice->Speak(speakContent, 0, NULL);

		pVoice->Release();

		pVoice = NULL;
	}

	//�ͷ�com��Դ
	::CoUninitialize();
}