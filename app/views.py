from django.shortcuts import render, redirect, reverse
    
def master(request):
    return render(request, 'master.html', {'test': True})