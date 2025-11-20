
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
typedef struct {
	char data;
	struct linknode* next;
}linknode, * linklist;
void reverse(linklist head) {
	linklist temp;
	int i = 0;
	linklist p = head;
	while (p != NULL) {
		i++;
		p = p->next;
	}
	for (int j = 0; j < i; j++) {
		for (int t = 0; t < (i - j); t++) {
			linklist q = p->next;
			if ((p->data) > (q->data)) {
				char temp = p->data;
				p->data = q->data;
				q->data = temp;
			}
			p = q;
		}
	}
}
