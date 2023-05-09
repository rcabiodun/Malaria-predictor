from django.db import models
from django.core.mail import send_mail
# Create your models here.
class EmailList(models.Model):
    email=models.EmailField(null=True,blank=True)

    def __str__(self):
        return self.email
    

class EmailMessages(models.Model):
    title=models.CharField(max_length=25,null=True,blank=True)
    message=models.TextField()

    def __str__(self):
        return self.message
    
    def save(self,*args,**kwargs):
        emails=EmailList.objects.all()
        email_list=[]
        for email in emails:
            email_list.append(str(email.email))
            #send messages to emails one at a time
        send_mail(
                    self.title,
                    self.message,
                    'Comfort',
                    email_list,
                )

        return super(EmailMessages,self).save(*args,**kwargs)