#include<stdio.h>
#include<string.h>
void sort(char* str) {
	char count[100] = { 0 };
	int len = strlen(str);
	for (int i = 0; i < len; i++) {
		count[str[i] - 'a']++;
	}
	int x[26];
	for (int i = 0; i < 26; i++) {
		x[i] = i;
	}
	for (int i = 0; i < 26; i++) {
		for (int j = i + 1; j < 26; j++) {
			if (count[i] < count[j]) {
				int temp1 = count[i];
				int temp2 = x[i];
				count[i] = count[j];
				count[j] = temp1;
				x[i] = x[j];
				x[j] = temp2;
			}
		}
	}
	for (int i = 0; i < 26; i++) {
		if (count[i] != 0) {
			char temp = x[i] + 'a';
			printf("%c:%d\n", temp, count[i]);
		}
	}

}
void main() {
	char str1[100];
	printf("ÇëÊäÈë×Ö·û´®:", str1);
	scanf_s("%s",str1,sizeof(str1) );
	sort(str1);
}
