#pragma once

#include "fs/xv6fs/defs.h"
#include "fs/xv6fs/file.h"
#include "sleeplock.h"

struct filesystem_type {
  const char *type;
  struct filesystem_operations *op;
};

#define DEVSIZ 32
#define MAXMNT 32
#define DIRSIZ 14
#define NPROC        64  // maximum number of processes
#define NCPU          8  // maximum number of CPUs
#define NOFILE       16  // open files per process
#define NFILE       100  // open files per system
#define NINODE       50  // maximum number of active i-nodes
#define NDEV         10  // maximum major device number
#define ROOTDEV       1  // device number of file system root disk
#define MAXARG       32  // max exec arguments
#define MAXOPBLOCKS  10  // max # of blocks any FS op writes
#define LOGSIZE      (MAXOPBLOCKS*3)  // max data blocks in on-disk log
#define NBUF         (MAXOPBLOCKS*3)  // size of disk block cache
#define FSSIZE       2000  // size of file system in blocks
#define MAXPATH      128   // maximum file path name
#define ROOTINO  1   // root i-number
#define BSIZE 1024  // block size
#define IBLOCK(i, sb)     ((i) / IPB + sb.inodestart)
#define O_RDONLY  0x000
#define O_WRONLY  0x001
#define O_RDWR    0x002
#define O_CREATE  0x200
#define O_TRUNC   0x400

// See https://unix.stackexchange.com/a/4403 for an explanation
// of superblocks, inodes, dentries and files.

struct super_block {
  struct filesystem_type *type;
  struct filesystem_operations *op;
  struct super_block *parent;
  struct inode *root;
  struct dentry *mountpoint;
  // Ignored at this stage.
  // This field records the mount device, i.e.,
  // the "/dev/sda" part in "mount /dev/sda".
  char device[DEVSIZ];
  struct super_block *mounts[MAXMNT];
  // FS-specific data for the mounted filesystem.
  // Usually we allocate a buffer and point sb->private to it.
  // The "void *private" fields in other structures are similar.
  void *private;
};

struct file {
  enum e_type type;
  struct filesystem_operations *op;
  // Reference count
  int ref;
  // read/write offset inside the file
  int off;
  char readable;
  char writable;
  struct inode *inode;
  void *private;
};

struct inode {
  struct filesystem_operations *op;
  // Which mounted fs does this inode belong to?
  struct super_block *sb;
  // Inode number
  uint inum;
  // Reference count (in memory)
  int ref;
  // protects everything below here
  struct sleeplock lock;
  short type;
  uint dev;
  uint size;
  short nlink;
  void *private;
};

#define DIRSIZ 14

struct dentry {
  struct filesystem_operations *op;
  // What is the parent directory? (Could be equal to de->inode.)
  struct inode *parent;
  char name[DIRSIZ];
  struct inode *inode;
  // Is this dentry a mount point?
  // Ignored at this stage.
  char ismount;
  // For an entry in the dentry cache, is the dentry already unlinked?
  char deleted;
  // Reference count
  int ref;
  void *private;
};

struct filesystem_operations {
  // Mount a filesystem.
  // Only used for the root mount at this stage.
  // Linux: file_system_type->mount
  struct super_block *(*mount) (const char *source);
  // Unmount a filesystem.
  // Ignored at this stage.
  // Linux: super_operations->umount_begin
  int (*umount) (struct super_block *sb);
  // Allocate an inode in the inode table on disk.
  // Linux: super_operations->alloc_inode
  struct inode *(*alloc_inode) (struct super_block *sb, short type);
  // Write (update) an existing inode.
  // Linux: super_operations->write_inode
  void (*write_inode) (struct inode *ino);
  // Called when the inode is recycled.
  // Linux: super_operations->evict_inode
  void (*release_inode) (struct inode *ino);
  // Free the inode in the inode table on disk.
  // Linux: super_operations->free_inode
  void (*free_inode) (struct inode *ino);
  // Truncate the file corresponding to inode.
  // Linux: (none)
  void (*trunc) (struct inode *ino);
  // Opens (returns a file instance) of the inode.
  // Linux: inode_operations->atomic_open
  struct file *(*open) (struct inode *ino, uint mode);
  // Closes an open file.
  // Linux: file_operations->flush
  void (*close) (struct file *f);
  // Reads from the file.
  // If dst_is_user==1, then dst is a user virtual address;
  // otherwise, dst is a kernel address.
  // Linux: file_operations->read
  int (*read) (struct inode *ino, char dst_is_user, uint64 dst, uint off, uint n);
  // Writes to the file.
  // Linux: file_operations->write
  int (*write) (struct inode *ino, char src_is_user, uint64 src, uint off, uint n);
  // Creates a new file.
  // target is a newly created dentry; target->inode is the actual file.
  // Linux: inode_operations->create
  int (*create) (struct inode *dir, struct dentry *target, short type, short major, short minor);
  // Creates a new link.
  // target is a newly created dentry; target->inode is the actual file.
  // Linux: inode_operations->link
  int (*link) (struct dentry *target);
  // Removes a link, and deletes a file if it is the last link.
  // Linux: inode_operations->unlink
  int (*unlink) (struct dentry *d);
  // look for a file in the directory.
  // Linux: inode_operations->lookup
  struct dentry *(*dirlookup) (struct inode *dir, const char *name);
  // Called when the dentry is recycled.
  // Linux: dentry_operations->d_release
  void (*release_dentry) (struct dentry *de);
  // Is the directory dp empty except for "." and ".." ?
  // Linux: (none)
  int (*isdirempty) (struct inode *dir);
  // initialize filesystem type
  // Linux: (none)
  void (*init) (void);

  void (*update_inode)(struct inode *ip);
};

// map major device number to device functions.
// struct devsw {
//   int (*read)(int, uint64, int);
//   int (*write)(int, uint64, int);
// };

//  extern struct devsw devsw[];

extern struct super_block *root;

extern struct filesystem_type xv6fs;

extern struct filesystem_type root_fs_type;

struct inode* idup(struct inode *ip);

void ilock(struct inode *ip);

void iunlock(struct inode *ip);

struct inode* nnamex(char *path, int nameiparent, char *name);

struct inode* namei(char *path);

int namecmp(const char *s, const char *t);

void iput(struct inode *ip);

void iunlockput(struct inode *ip);

struct inode* inget(uint dev, uint inum);

char* nskipelem(char *path, char *name);

struct file* filedup(struct file* f);

struct file* filealloc(void);

int fileread(struct file* f, uint64 addr, int n);

int filewrite(struct file* f, uint64 addr, int n);

void stati(struct inode *ip, struct stat *st);

int filestat(struct file* f, uint64 addr);

struct inode* nameiparent(char *path, char *name);

int dirlink(struct inode *dp, char *name, uint inum) ;

void iinit();


