#include<iostream>
#include<string.h>
#include<stdio.h>
char reverse(char p[], int i, int j) {
	char temp;
	temp = p[i];
	p[j] = p[i];
	p[j] = temp;
	if (i < j)
		reverse(p, i++, j--);
}
void main() {
	printf("ÇëÊäÈë×Ö·û´®");
	char p[100] = { 0 };
	scanf("%s", p);
	int i = 0, j = strlen(p);
	reverse(p, i, j);
	puts(p);
}