ls .\ -r | where {$_.extension -eq ".jpg"} | % { $_.FullName.substring($pwd.Path.length+1) + $(if($_.PsIsContainer){'\'}) } | Out-File dataset.txt -Encoding ASCII