#include "vfs.h"
#include "defs.h"
#include "fs/xv6fs/file.h"
#include "stat.h"
#include "proc.h"

struct {
  struct inode inode[NINODE];
} intable;

struct {
  struct file file[NFILE];
} fntable;

struct super_block* root;

struct inode* idup(struct inode *ip){
 // printf("idup\n");
    ip->ref++;
    return ip;
}

void iunlock(struct inode *ip){
 // printf("iunlock\n");
  if(ip == 0 || !holdingsleep(&ip->lock) || ip->ref < 1){
    panic("iunlock");
  }
    
  releasesleep(&ip->lock);
 // printf("iunlock complete\n");
}

struct inode* nnamex(char *path, int nameiparent, char *name) {
  //printf("nnamex\n");
    struct inode *ip, *next;

    if(*path == '/'){
      if(root==0) ip = inget(ROOTDEV, ROOTINO);
      else ip=root->root;
    }
    else
        ip = idup(myproc()->cwd);

    while((path = nskipelem(path, name)) != 0){
        ilock(ip);
        if(ip->type != T_DIR){
            iunlockput(ip);
      //  printf("nnamex complete\n");
        return 0;
    }
    if(nameiparent && *path == '\0'){
      // Stop one level early.
      iunlock(ip);
    //  printf("nnamex complete\n");
      return ip;
    }
    if((next = ip->op->dirlookup(ip, name)->inode) == 0){
      iunlockput(ip);
    //  printf("nnamex complete\n");
      return 0;
    }
    iunlockput(ip);
    ip = next;
  }
  if(nameiparent){
    iput(ip);
//    printf("nnamex complete\n");
    return 0;
  }
//  printf("nnamex complete\n");
  return ip;
}

struct inode* namei(char *path){
//  printf("namei\n");
    char name[DIRSIZ];
 //   printf("namei complete\n");
    return nnamex(path, 0, name);
}

int namecmp(const char *s, const char *t){
  return strncmp(s, t, DIRSIZ);
}

void iunlockput(struct inode *ip) {
 // printf("iunlockput\n");
  iunlock(ip);
  iput(ip);
 // printf("iunlockput complete\n");
}

struct inode* inget(uint dev, uint inum)
{
 // printf("inget\n" );
  struct inode *ip, *empty;

  // Is the inode already in the table?
  empty = 0;
  for(ip = &intable.inode[0]; ip < &intable.inode[NINODE]; ip++){
    if(ip->ref > 0 && ip->dev == dev && ip->inum == inum){
      ip->ref++;
      return ip;
    }
    if(empty == 0 && ip->ref == 0)    // Remember empty slot.
      empty = ip;
  }

  // Recycle an inode entry.
  if(empty == 0)
    panic("inget: no inodes");

  ip = empty;
  ip->dev = dev;
  ip->inum = inum;
  ip->ref = 1;
  ip->op = xv6fs.op;
  ip->private = 0;
//  printf("inget complete\n");
  return ip;
}

char* nskipelem(char *path, char *name)
{
 // printf("nskipelem\n");
  char *s;
  int len;

  while(*path == '/')
    path++;
  if(*path == 0)
    return 0;
  s = path;
  while(*path != '/' && *path != 0)
    path++;
  len = path - s;
  if(len >= DIRSIZ)
    memmove(name, s, DIRSIZ);
  else {
    memmove(name, s, len);
    name[len] = 0;
  }
  while(*path == '/')
    path++;
 // printf("nskipelem complete\n");
  return path;
}

struct file* filedup(struct file* f){
 // printf("filedup\n");
  if(f->ref < 1)
    panic("filedup");
  f->ref++;
 // printf("filedup complete\n");
  return f;
}

struct file* filealloc(void) {
 // printf("filealloc\n");
  struct file *f;

  for(f = fntable.file; f < fntable.file + NFILE; f++){
    if(f->ref == 0){
      f->ref = 1;
      return f;
    }
  }
 // printf("filealloc complete\n");
  return 0;
}

int fileread(struct file* f, uint64 addr, int n) {
 // printf("fileread\n");
  int r = 0;
  if(f->readable == 0)
    return -1;
  if(f->type == FD_DEVICE)
    return devsw[CONSOLE].read(1, addr, n);
  ilock(f->inode);
  if((r = f->inode->op->read(f->inode, 1, addr, f->off, n)) > 0)
    f->off += r;
  iunlock(f->inode);
 // printf("fileread complete\n");
  return r;
}

int filewrite(struct file* f, uint64 addr, int n) {
 // printf("filewrite\n");
  int r, ret = 0;

  if(f->writable == 0)
    return -1;

  if(f->type == FD_DEVICE)
    return devsw[CONSOLE].write(1, addr, n);

  int max = ((MAXOPBLOCKS-1-1-2) / 2) * BSIZE;
  int i = 0;
  while(i < n){
    int n1 = n - i;
    if(n1 > max)
      n1 = max;

    ilock(f->inode);
    if ((r = f->inode->op->write(f->inode, 1, addr + i, f->off, n1)) > 0)
      f->off += r;
    iunlock(f->inode);

    if(r != n1){
        // error from writei
      break;
    }
    i += r;
    }
    ret = (i == n ? n : -1);
  //  printf("filewrite complete\n");
  return ret;
}

void stati(struct inode *ip, struct stat *st)
{
  printf("stati\n");
  st->dev = ip->dev;
  st->ino = ip->inum;
  st->type = ip->type;
  st->nlink = ip->nlink;
  st->size = ip->size;
//  printf("stati complete\n");
}

int filestat(struct file* f, uint64 addr){
  printf("filestat\n");
  struct proc *p = myproc();
  struct stat st;

  if(f->type != FD_INODE && f->type != FD_DEVICE) return -1;
  ilock(f->inode);
  stati(f->inode, &st);
  iunlock(f->inode);
  if(copyout(p->pagetable, addr, (char *)&st, sizeof(st)) < 0)
    return -1;
 // printf("filestat complete\n");
  return 0;
}

struct inode* nameiparent(char *path, char *name) {
 // printf("nameiparent\n");
  return nnamex(path, 1, name);
}


int dirlink(struct inode *dp, char *name, uint inum) {
 // printf("dirlink\n");
  struct dentry *de;

  de = (struct dentry *)kalloc();
  de->private = (uint *)kalloc();
  de->parent = dp;
  *(uint *)de->private = inum;

  for(int i=0;i<DIRSIZ;i++){
    de->name[i] = name[i];
  }

  int ret = dp->op->link(de);

  return ret;

  // if((ip = dirlookup(dp, name, 0)) != 0){
  //   iput(ip);
  //   return -1;
  // }

  // for(off = 0; off < dp->size; off += sizeof(de)){
  //   if(dp->op->read(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
  //     panic("dirlink read");
  //   if(de.inode->inum == 0)
  //     break;
  // }

  // strncpy(de.name, name, DIRSIZ);
  // de.inode->inum = inum;
  // if(dp->op->write(dp, 0, (uint64)&de, off, sizeof(de)) != sizeof(de))
  //   return -1;
  // printf("dirlink complete\n");
  return 0;
}

void ilock(struct inode *ip){
//  printf("ilock\n");
  if(ip == 0 || ip->ref < 1)
    panic("ilock");
  acquiresleep(&ip->lock);
  if(ip->private == 0){
    ip->op->update_inode(ip);
  }
 // printf("ilock complete");
}

void iput(struct inode *ip){
 // printf("iput\n");
  if(ip->ref == 1 && ip->private && ip->nlink == 0){
    acquiresleep(&ip->lock);

    ip->op->trunc(ip);
    ip->type = 0;
    ip->op->write_inode(ip);
    kfree(ip->private);
    ip->private = 0;

    releasesleep(&ip->lock);
  }

  ip->ref--;
 // printf("iput complete\n");
}

void iinit() {
 // printf("iinit\n");
  int i = 0;
  
  for(i = 0; i < NINODE; i++) {
    initsleeplock(&intable.inode[i].lock, "inode");
  }
 // printf("iinit complete\n");
}