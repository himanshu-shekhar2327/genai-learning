from pydantic import BaseModel , EmailStr , Field
from typing import Optional


class Student(BaseModel):
    name : str
    age : Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10) # can apply contraint

# new_student = {'name':32}  # This gives Validation Error
new_student = {'name':'Ram'} # this gives name : ram 
age: Optional[int] = None

new_student = {'age':'32', 'email' : 'abc@gmail.com'}
new_student = {'age':'32', 'email' : 'abccom'} # value is not a valid email address: An email address must have an @-sign.





student = Student(**new_student)
print(student)
print(student.name)