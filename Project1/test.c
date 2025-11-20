#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>

void delet(char *str1,char *str2 ) {
	char* p1, * p2;
	char result[100] = {0};
	p2 = str1;
	while ((p1 = strstr(str1, str2)) != NULL) {
		strncat(result, str1, p1 - p2);
		strcpy(&str1, p1 + strlen(str2));
		p2 = str1;
	}
	strcat(result, str1);
	printf("删除后主串为%s", result);

}
void main() {
	char str1[10] = {0};
	char str2[10] = {0};
	printf("请输入要删除的主串:");
	scanf("%s", str1);

	printf("请输入要删除的子串:");
	scanf("%s", str2);

	delet(str1, str2);
}